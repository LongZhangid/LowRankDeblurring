import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

class LowRankDeblur:
    """
    Image deblurring using low-rank approximation.
    Problem:
        X_hat = argmin_X {λ/2 * || K * X - F ||_2^2 + phi(X)}
    """
    def __init__(self, F, H, ref, eta, 
                 beta:float=1.25, c0:float=0.4, c1=1.25,
                 patch_size:int=4, similar_num:int=20,
                 search_size:int=21, step=3, max_iter=35, tol=1e-6):
        self.m, self.n = F.shape
        self.H = H
        self.F = F
        self.ref = ref
        self.eta = eta      # noise standard deviation
        self.beta = beta    # beta > 1, parameter for updating lambda
        self.c0 = c0        # parameters for noise estimation
        self.c1 = c1        # parameters for noise estimation and thresholding
        self.patch_size = patch_size    # patch size
        self.similar_num = similar_num  # number of similar patches
        self.search_size = search_size  # search window size
        self.step = step          # step size for reference patches
        self.max_iter = max_iter    # maximum number of iterations
        self.tol = tol         # tolerance for stopping criterion

    def _solver(self):
        """
        Solve the image deblurring problem using low-rank approximation.
        """
        # Initialize variables
        self.fH = np.conj(self._FFT(self.H, self.F.shape))
        self.fHtH = np.abs(self.fH)**2 
        self.fHtF = self.fH * self._FFT(self.F)
        
        self.X = self.F.copy()  # Initial estimate
        
        # Initialize lambda_ with care to avoid negative denominator
        mean_val = np.mean(self.F)
        sqrd_norm = np.sum((self.F - mean_val)**2)
        self.lambda_ = (self.m * self.n * self.eta**2) / (sqrd_norm - self.m * self.n * self.eta**2 + 1e-10)

        self.sigma_s = self.eta  # Now storing standard deviation instead of variance
        self.psnrl = []
        self.ssiml = []
        self.ssiml.append(ssim(self.ref, self.F, data_range=self.ref.max() - self.ref.min()))
        self.psnrl.append(psnr(self.ref, self.F, data_range=self.ref.max() - self.ref.min()))
        print(f"Initial PSNR: {self.psnrl[0]:.4f} dB, SSIM: {self.ssiml[0]:.4f}")
        
        # Precompute the frequency response terms for noise estimation
        self.H1 = self.fH / (self.fHtH + self.lambda_)
        self.H2 = self.lambda_ / (self.fHtH + self.lambda_)
        
        for i in range(self.max_iter):
            X_prev = self.X.copy()
            self._sub_deblur()      # Update Y
            self._noise_estimate()  # Update sigma_s
            self._sub_denoise()     # Update X
            self._update_lambda_()  # Update lambda
            
            # Update frequency response terms for next iteration
            self.H1 = self.fH / (self.fHtH + self.lambda_)
            self.H2 = self.lambda_ / (self.fHtH + self.lambda_)
            # Compute PSNR and SSIM
            self.ssiml.append(ssim(self.ref, self.X, data_range=self.ref.max() - self.ref.min()))
            self.psnrl.append(psnr(self.ref, self.X, data_range=self.ref.max() - self.ref.min()))
            if (i+1) % 1 == 0:
                print(f"Iteration {i+1:3d}, PSNR: {self.psnrl[i+1]:.4f} dB, SSIM: {self.ssiml[i+1]:.4f}")
            if np.linalg.norm(self.X - X_prev) / np.linalg.norm(X_prev + 1e-10) < self.tol:
                break
        return self.X, self.psnrl, self.ssiml
    
    def _FFT(self, X, shape=None):
        """
        Compute the FFT of an object.
        """
        if shape is not None:
            return np.fft.fft2(X, s=shape)
        else:
            return np.fft.fft2(X)
        
    def _IFFT(self, X):
        """
        Compute the inverse FFT of an object.
        """
        return np.fft.ifft2(X).real
    
    def _sub_deblur(self):
        """
        Subroutine for image deblurring:
        y^{(k)} = argmin_y {(1/2)||H*y - g||^2 + (λ^{(k)}/2)||y - x^{(k-1)}||^2}
        """
        fX = self._FFT(self.X)
        fY = (self.fHtF + self.lambda_ * fX) / (self.fHtH + self.lambda_)
        self.Y = self._IFFT(fY)
    
    def _noise_estimate(self):
        """
        Estimate the noise variance in the image.
        """
        # Calculate frequency domain filter responses
        Hg = self.fH / (self.fHtH + self.lambda_ + 1e-10)
        Hh = self.lambda_ / (self.fHtH + self.lambda_ + 1e-10)
        
        # Calculate noise variance contributions
        sig_g = np.sum(np.abs(Hg.flatten())**2) / (self.m * self.n)
        sig_h = np.sum(np.abs(Hh.flatten())**2) / (self.m * self.n)
        
        # Calculate difference
        dif = self.Y - self.X
        vd = self.sigma_s**2 - np.mean(dif**2)
        
        # Estimate noise standard deviation (not variance)
        self.sigma_s = np.sqrt(np.abs(vd) * sig_h * self.c0 + self.eta**2 * sig_g) * self.c1
        
    def _block_match(self, ref_row, ref_col):
        """
        Block matching - find similar image patches
        
        Parameters:
            image: input image
            ref_row, ref_col: reference patch position
            
        Returns:
            similar_blocks: list of indices of similar patches
        """
        # Define reference patch
        ref_block = self.Y[ref_row:ref_row+self.patch_size, ref_col:ref_col+self.patch_size]
        
        # Determine search range
        r_min = max(0, ref_row - self.search_size)
        r_max = min(self.m - self.patch_size, ref_row + self.search_size)
        c_min = max(0, ref_col - self.search_size)
        c_max = min(self.n - self.patch_size, ref_col + self.search_size)
        
        # Calculate distances between reference patch and all possible patches
        distances = []
        blocks = []
        
        for i in range(r_min, r_max + 1):
            for j in range(c_min, c_max + 1):
                block = self.Y[i:i+self.patch_size, j:j+self.patch_size]
                dist = np.mean((block - ref_block)**2)
                distances.append(dist)
                blocks.append((i, j))
        
        # Select the most similar nblk patches
        indices = np.argsort(distances)[:self.similar_num]
        similar_blocks = [blocks[i] for i in indices]
        
        return similar_blocks
    
    def _low_rank_denoise(self, grouped_patches):
        """
        Perform low-rank denoising on the image patches.
        Based on the second code's implementation
        """
        mean_patches = np.mean(grouped_patches, axis=1, keepdims=True)
        centered_patches = grouped_patches - mean_patches
        try:
            U, Sigma, Vt = np.linalg.svd(centered_patches, full_matrices=False)
        except:
            # If SVD fails, return original matrix
            return grouped_patches, 1.0, grouped_patches.shape[1]
        
        # Calculate adaptive threshold (sigma_s is standard deviation)
        S = np.maximum(Sigma**2 / grouped_patches.shape[1] - self.sigma_s**2, 0)
        threshold = self.c1 * self.sigma_s**2 / (np.sqrt(S) + 1e-10)
        
        # Soft thresholding
        Sigma_hat = np.maximum(Sigma - threshold, 0)
        
        # Calculate rank
        rank = np.sum(Sigma_hat > 0)
        
        # Reconstruct matrix
        if rank > 0:
            X_hat_centered = U[:, :rank] @ np.diag(Sigma_hat[:rank]) @ Vt[:rank, :]
        else:
            X_hat_centered = np.zeros_like(centered_patches)
        
        # Add back mean
        X_hat = X_hat_centered + mean_patches
        
        # Calculate weight based on rank
        weight = (grouped_patches.shape[0] - rank) / grouped_patches.shape[0]
        
        return X_hat, weight, rank

    def _sub_denoise(self):
        """
        Subroutine for image denoising using low-rank approximation.
        x^{(k)} = argmin_{x} { (λ^{(k)}/2) ||y^{(k)} - x||^2 + Φ(x) }
        """
        # Create a temporary image for aggregation
        temp_img = np.zeros_like(self.X)
        weight_img = np.zeros_like(self.X)
        
        # Process each group of similar patches
        for i in range(0, self.m - self.patch_size + 1, self.step):
            for j in range(0, self.n - self.patch_size + 1, self.step):
                similar_blocks = self._block_match(i, j)
                grouped_patches = np.zeros((self.patch_size**2, len(similar_blocks)))
                for idx, (r, c) in enumerate(similar_blocks):
                    block = self.Y[r:r+self.patch_size, c:c+self.patch_size]
                    grouped_patches[:, idx] = block.flatten()
                # Low-rank denoising
                denoised_patches, weight, _ = self._low_rank_denoise(grouped_patches)
                
                # Aggregate denoised patches back to the image
                for idx, (r, c) in enumerate(similar_blocks):
                    patch = denoised_patches[:, idx].reshape(self.patch_size, self.patch_size)
                    temp_img[r:r+self.patch_size, c:c+self.patch_size] += patch * weight
                    weight_img[r:r+self.patch_size, c:c+self.patch_size] += weight
        # Avoid division by zero
        weight_img[weight_img == 0] = 1e-3
        self.X = temp_img / weight_img
    
    def _update_lambda_(self):
        """
        Update the penalty parameter λ.
        λ^{(k+1)} = β * λ^{(k)}
        """
        self.lambda_ *= self.beta


if __name__ == "__main__":
    # Example usage
    img = Image.open("colors.png").convert("L")
    img = np.array(img)

    # Simulate a blur kernel
    kernel_size = 25
    sigma = 1.6  # Use a smaller sigma for Gaussian kernel
    ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel /= np.sum(kernel)

    # Simulate a blurred and noisy image
    fimg = np.fft.fft2(img)
    H = np.fft.fft2(kernel, s=img.shape)
    blurred = np.fft.ifft2(fimg * H).real
    noise = 0.9 * np.random.randn(img.shape[0], img.shape[1])
    noisy_blurred = blurred + noise
    
    # Use smaller parameters for testing
    deblurrer = LowRankDeblur(noisy_blurred, kernel, img, 
                              eta=0.9, max_iter=100)
    restored_img, snrl, ssiml = deblurrer._solver()
    
    # Plot results
    plt.figure(figsize=(12, 4))
    plt.subplot(2, 3, 1)
    plt.title("Original Image")
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.title("Noisy Blurred Image")
    plt.imshow(noisy_blurred, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.title("Restored Image")
    plt.imshow(restored_img, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.title("SNR over Iterations")
    plt.plot(snrl)
    plt.xlabel("Iteration")
    plt.ylabel("SNR (dB)")

    plt.subplot(2, 3, 5)
    plt.title("SSIM over Iterations")
    plt.plot(ssiml)
    plt.xlabel("Iteration")
    plt.ylabel("SSIM")
    
    plt.tight_layout()
    plt.show()