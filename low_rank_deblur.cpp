/*
By Long Zhang in CIOMP
*/


#define cimg_use_png

#include <iostream>
#include <fftw3.h>
#include <Eigen/Dense>
#include <matplot/matplot.h>
#include <CImg.h>
#include <complex>
#include <cmath>
#include <algorithm>

namespace cimg = cimg_library;
namespace plt = matplot;


Eigen::VectorXcd fft_img(Eigen::MatrixXd& img, int height, int width);
Eigen::MatrixXd ifft_img(Eigen::VectorXcd& freq, int height, int width);

class LowRankDeblur {
private:
	Eigen::MatrixXd F, H, ref, X;
	double eta, beta, c0, c1, tol;
	int patch_size, search_size, similar_num, step, max_iter;

	int m, n;
	Eigen::VectorXcd fHtF, fH;
	Eigen::MatrixXd fHtH, Y;
	double lambda, sigma_s;
	std::vector<double> psnrl;

	Eigen::VectorXcd fft2(Eigen::MatrixXd& img) {
		/*
		2 dimension FFT of image.
		*/
		Eigen::MatrixXd padded_img = Eigen::MatrixXd::Zero(m, n);
		if (img.rows() != m || img.cols() != n) {
			int r = m - img.rows(), c = n - img.cols();
			int start_r = r / 2, start_c = c / 2;
			for (int i = 0; i < img.rows(); i++) {
				for (int j = 0; j < img.cols(); j++) {
					padded_img(start_r + i, start_c + j) = img(i, j);
				}
			}
		}
		else {
			padded_img = img;
		}
		fftw_complex* in = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * n * m));
		fftw_complex* out = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * n * m));
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				in[i * n + j][0] = padded_img(i, j);
				in[i * n + j][1] = 0;
			}
		}
		fftw_plan plan = fftw_plan_dft_2d(m, n, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
		fftw_execute(plan);
		Eigen::VectorXcd result(m * n);
		for (int i = 0; i < m * n; i++) {
			result(i) = std::complex<double>(out[i][0], out[i][1]);
		}
		fftw_destroy_plan(plan);
		fftw_free(in);
		fftw_free(out);
		return result;
	}

	Eigen::MatrixXd ifft2(Eigen::VectorXcd& freq) {
		int N = m * n;
		fftw_complex* in = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * N));
		fftw_complex* out = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * N));
		for (int i = 0; i < N; i++) {
			in[i][0] = freq[i].real();
			in[i][1] = freq[i].imag();
		}
		fftw_plan plan = fftw_plan_dft_2d(m, n, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
		fftw_execute(plan);
		Eigen::MatrixXd result(m, n);
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				result(i, j) = out[i * n + j][0] / (m * n);
			}
		}
		return result;
	}

	void sub_deblur() {
		Eigen::VectorXcd fX = fft2(X);
		Eigen::VectorXcd fY = (fHtF.array() + lambda * fX.array()) / (fHtH.array() + lambda);
		Y = ifft2(fY);
	}

	void noise_estimate() {
		Eigen::MatrixXcd Hg = fH.array() / (fHtH.array() + lambda);
		Eigen::MatrixXcd Hh = lambda / (fHtH.array() + lambda);
		double sig_g = Hg.array().abs2().mean();
		double sig_h = Hh.array().abs2().mean();
		Eigen::MatrixXd dif = Y - X;
		double vd = std::pow(sigma_s, 2) - dif.array().abs2().mean();
		sigma_s = c1 * std::sqrt(std::abs(vd) * sig_h * c0 + std::pow(eta, 2) * sig_g);
	}

	std::vector<std::pair<int, int>> block_match(int ref_row, int ref_col) {
		/*
				Block matching - find similar image patches

		Parameters:
			image: input image
			ref_row, ref_col: reference patch position

		Returns:
			similar_blocks: list of indices of similar patches
		*/
		Eigen::MatrixXd ref_block = Y.block(ref_row, ref_col, patch_size, patch_size);

		// Determine search range
		int r_min = std::max<int>(0, ref_row - search_size);
		int r_max = std::min<int>(m - patch_size, ref_row + search_size);
		int c_min = std::max<int>(0, ref_col - search_size);
		int c_max = std::min<int>(n - patch_size, ref_col + search_size);

		std::vector<double> distances;
		std::vector<std::pair<int, int>> blocks;
		for (int i = r_min; i < r_max + 1; i++) {
			for (int j = c_min; j < c_max + 1; j++) {
				Eigen::MatrixXd block = Y.block(i, j, patch_size, patch_size);
				double dist = (block - ref_block).squaredNorm() / (patch_size * patch_size);
				distances.push_back(dist);
				blocks.push_back({ i, j });
			}
		}
		std::vector<size_t> indices(distances.size());
		std::iota(indices.begin(), indices.end(), 0);
		std::partial_sort(indices.begin(), indices.begin() + similar_num, indices.end(),
			[&distances](size_t left, size_t right) -> bool {
				return distances[left] < distances[right];
			});
		std::vector <std::pair<int, int>> result;
		for (int i = 0; i < distances.size(); i++) {
			result.push_back(blocks[indices[i]]);
		}
		return result;
	}

	std::pair<Eigen::MatrixXd, double> low_rank_denoise(Eigen::MatrixXd grouped_patches) {
		/*
		Perform low-rank denoising on the image patches.
		Based on the second code's implementation
		*/
		double mean_patches = grouped_patches.mean();
		Eigen::MatrixXd centered_patchs = grouped_patches.array() - mean_patches;
		Eigen::MatrixXd U, Vt;
		Eigen::VectorXd Sigma;
		Eigen::JacobiSVD<Eigen::MatrixXd> svd(
			centered_patchs, Eigen::ComputeThinU | Eigen::ComputeThinV
		);
		U = svd.matrixU();
		Vt = svd.matrixV().transpose();
		Sigma = svd.singularValues();
		Eigen::VectorXd S = (Sigma.array().square() / grouped_patches.cols() - sigma_s * sigma_s).cwiseMax(0.0);
		Eigen::VectorXd threshold = c1 * sigma_s * sigma_s / (Sigma.array().sqrt() + 1e-10);
		Eigen::VectorXd Sigma_hat = Sigma.array().sign() * (Sigma.array().abs() - threshold.array()).cwiseMax(0.0);

		// Calculate rank
		int rank = std::count_if(Sigma.begin(), Sigma.end(),
			[](double value) {return value > 0.0; });
		Eigen::MatrixXd X_hat_centered = Eigen::MatrixXd::Zero(patch_size * patch_size, similar_num);
		if (rank > 0) {
			Eigen::MatrixXd U_rank = U.block(0, 0, U.rows(), rank);
			Eigen::MatrixXd Vt_rank = Vt.block(0, 0, rank, Vt.cols());
			Eigen::VectorXd Sigma_rank = Sigma_hat.head(rank);
			Eigen::MatrixXd Sigma_diag = Sigma_rank.asDiagonal();
			X_hat_centered = U_rank * Sigma_diag * Vt_rank;
		}
		Eigen::MatrixXd X_hat = X_hat_centered.array() + mean_patches;
		// Calculate weight based on rank
		double weight = (grouped_patches.rows() - rank) / grouped_patches.rows();
		return std::make_pair(X_hat, weight);
	}

	void sub_denoise() {
		/*
		Subroutine for image denoising using low-rank approximation.
		x^{(k)} = argmin_{x} { (¦Ë^{(k)}/2) ||y^{(k)} - x||^2 + ¦µ(x) }
		*/
		Eigen::MatrixXd temp_img = Eigen::MatrixXd::Zero(m, n);
		Eigen::MatrixXd weight_img = Eigen::MatrixXd::Zero(m, n);

		for (int i = 0; i < m - patch_size + 1; i += step) {
			for (int j = 0; j < n - patch_size + 1; j += step) {
				std::vector<std::pair<int, int>> similar_blocks = block_match(i, j);
				Eigen::MatrixXd grouped_patches =
					Eigen::MatrixXd::Zero(patch_size * patch_size, similar_blocks.size());
				int idx = 0;
				for (auto block_pos : similar_blocks) {
					int row = block_pos.first;
					int col = block_pos.second;
					Eigen::MatrixXd block = Y.block(row, col, patch_size, patch_size);
					for (int x = 0; x < patch_size; x++) {
						for (int y = 0; y < patch_size; y++) {
							grouped_patches(x * patch_size + y, idx) = block(x, y);
						}
					}
					idx++;
				}
				Eigen::MatrixXd denoised_patches;
				double weight;
				std::tie(denoised_patches, weight) = low_rank_denoise(grouped_patches);

				for (size_t idx = 0; idx < similar_blocks.size(); idx++) {
					auto pos = similar_blocks[idx];
					int r = pos.first;
					int c = pos.second;
					
					for (size_t xx = 0; xx < patch_size; xx++) {
						for (size_t yy = 0; yy < patch_size; yy++) {
							double value = denoised_patches(xx * patch_size + yy, idx);
							temp_img(r + xx, c + yy) = value * weight;
							weight_img(r + xx, c + yy) = weight;
						}
					}
				}
			}
		}
		// Avoid division by zero
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				if (weight_img(i, j) < 1e-10) {
					weight_img(i, j) = 1e-10;
				}
			}
		}
		X = temp_img.array() / weight_img.array();
	}

	void update_lambda() {
		/*
		Update the penalty parameter ¦Ë.
		¦Ë^{(k+1)} = ¦Â * ¦Ë^{(k)}
		*/
		lambda *= beta;
	}

	double psnr() {
		/*
		Calculate PSNR
		*/
		return 10 * std::log10((ref.array() * ref.array()).maxCoeff() / (ref - X).array().square().mean());
	}


public:
	// Constructor
	LowRankDeblur(Eigen::MatrixXd F, Eigen::MatrixXd H,
		const Eigen::MatrixXd ref, double eta, double beta = 1.25,
		double c0 = 0.4, double c1 = 1.25, double tol = 1e-5,
		int patch_size = 4, int search_size = 21, int similar_num = 20,
		int step = 3, int max_iter = 35) : F(F), H(H), ref(ref),
		eta(eta), beta(beta), c0(c0), c1(c1), tol(tol), patch_size(patch_size),
		search_size(search_size), similar_num(similar_num),
		step(step), max_iter(max_iter) {
		m = F.rows();
		n = F.cols();
		X = F;
		fH = fft2(H);
		fHtF = fH.array() * fft2(F).array();
		fHtH = fH.cwiseAbs2();
		double mean_val = F.mean();
		double sqrd_norm = (F.array() - mean_val).cwiseAbs2().sum();
		lambda = (m * n * eta * eta) / (sqrd_norm - m * n * eta * eta + 1e-10);
		sigma_s = eta;
	}

	Eigen::MatrixXd solve() {
		/*
		Solve the image deblurring problem using low-rank approximation.
		*/
		std::cout << "Start low rank deblurring ..." << std::endl;
		psnrl.push_back(psnr());
		std::cout << "Initial PSNR: " << psnrl[0] << std::endl;
		Eigen::MatrixXd X_pre = F;
		for (size_t i = 0; i < max_iter; i++) {
			sub_deblur();
			noise_estimate();
			sub_denoise();
			update_lambda();
			psnrl.push_back(psnr());
			if ((i + 1) % 1 == 0) {
				std::cout << "Iteration " << i + 1 << ", PSNR: " << psnrl[i + 1] << "\n";
			}
			if ((i + 1) >= max_iter) {
				break;
			}
		}
		return X;
	}

	std::vector<double> get_estimation() {
		/*
		Get PSNR and SSIM.
		*/
		return psnrl;
	}
};


int main() {
	cimg::CImg<double> ref_img("colors.png");
	if (ref_img.spectrum() > 1) {
		ref_img = ref_img.get_channel(0);
	}
	// Display original image.
	cimg::CImgDisplay orig_dis(ref_img, "Original Image", 0);
	ref_img.save("original_image.png");
	int height = ref_img.height();
	int width = ref_img.width();
	// Create blur kernel (25x25 Gaussian)
	Eigen::MatrixXd H(25, 25);
	double sigma = 1.0;
	double sum = 0.0;
	for (int i = 0; i < 25; i++) {
		for (int j = 0; j < 25; j++) {
			double x = i - 2;
			double y = j - 2;
			H(j, i) = exp(-(x * x + y * y) / (2 * sigma * sigma));
			sum += H(j, i);
		}
	}
	H /= sum;  // Normalize kernel

	// Attain blurring and addition noise image.
	Eigen::MatrixXd ref(height, width);
	for (size_t i = 0; i < height; i++) {
		for (size_t j = 0; j < width; j++) {
			ref(i, j) = ref_img(i, j);
		}
	}

	Eigen::VectorXcd fH = fft_img(H, height, width);
	Eigen::VectorXcd fref = fft_img(ref, height, width);
	Eigen::VectorXcd freq = fH.array() * fref.array();
	Eigen::MatrixXd F = ifft_img(freq, height, width);

	// Display blurring image.
	cimg::CImg<double> F_img(height, width);
	for (size_t i = 0; i < height; i++) {
		for (size_t j = 0; j < width; j++) {
			F_img(i, j) = F(i, j);
		}
	}
	cimg::CImgDisplay blur_dis(F_img, "Blurring image", 0);
	F_img.save("blurring_image.png");

	// Deblur processing
	LowRankDeblur low_rank_deblur(F, H, ref, 1.6);
	Eigen::MatrixXd X = low_rank_deblur.solve();

	// Display result
	cimg::CImg<double> result(height, width);
	for (size_t i = 0; i < height; i++) {
		for (size_t j = 0; j < width; j++) {
			result(i, j) = X(i, j);
		}
	}
	cimg::CImgDisplay deblur_dis(result, "Deblurring Image", 0);
	result.save("deblurring_image.png");

	// Get estimation factor
	std::vector<double> psnrl = low_rank_deblur.get_estimation();

	// Display PSNR
	std::vector<double> iterations(psnrl.size());
	std::iota(iterations.begin(), iterations.end(), 0);

	plt::figure();
	plt::plot(iterations, psnrl);
	plt::title("PSNR Convergence");
	plt::xlabel("Iteration");
	plt::ylabel("PSNR");
	plt::grid(true);
	plt::save("psnr_convergence.png");
	plt::show();

	return 0;
}


Eigen::VectorXcd fft_img(Eigen::MatrixXd& img, int height, int width) {
	/*
	2 dimension FFT of image.
	*/
	int m = height;
	int n = width;
	Eigen::MatrixXd padded_img = Eigen::MatrixXd::Zero(m, n);
	if (img.rows() != m && img.cols() != n) {
		int r = m - img.rows(), c = n - img.cols();
		int start_r = r / 2, start_c = c / 2;
		for (int i = 0; i < img.rows(); i++) {
			for (int j = 0; j < img.cols(); j++) {
				padded_img(start_r + i, start_c + j) = img(i, j);
			}
		}
	}
	else {
		padded_img = img;
	}
	fftw_complex* in = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * n * m));
	fftw_complex* out = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * n * m));
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			in[i * n + j][0] = padded_img(i, j);
			in[i * n + j][1] = 0;
		}
	}
	fftw_plan plan = fftw_plan_dft_2d(m, n, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(plan);
	Eigen::VectorXcd result(m * n);
	for (int i = 0; i < m * n; i++) {
		result(i) = std::complex<double>(out[i][0], out[i][1]);
	}
	fftw_destroy_plan(plan);
	fftw_free(in);
	fftw_free(out);
	return result;
}

Eigen::MatrixXd ifft_img(Eigen::VectorXcd& freq, int m, int n) {
	int N = m * n;
	fftw_complex* in = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * N));
	fftw_complex* out = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * N));
	for (int i = 0; i < N; i++) {
		in[i][0] = freq[i].real();
		in[i][1] = freq[i].imag();
	}
	fftw_plan plan = fftw_plan_dft_2d(m, n, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
	fftw_execute(plan);
	Eigen::MatrixXd result(m, n);
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			result(i, j) = out[i * n + j][0] / (m * n);
		}
	}
	return result;

}
