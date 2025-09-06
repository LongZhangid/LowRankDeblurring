# LowRankDeblurring

Now, we have achieved the program of image deblurring using low rank method:

X_hat = argmin_X {λ/2 || K * X - F ||_2^2 + phi(X)}

Where, K is PSF, X is ideal image and F is imaging via optical system, phi(x) is prior term.

The result (Here, parameters is not adjustive, if you want to use it, please to do it.):

<img width="1366" height="663" alt="Figure_1" src="https://github.com/user-attachments/assets/7e53b3a2-9c54-4aa7-a225-2d4ca2f33c1c" />

If you want to attain more detailed information, you can read file for LowRankApproximation.pdf

Update in 2025/09/06

Now, We update c++ file that accomplish same function.

# =====================================
# Non-Local Image Denoising Algorithm

## 1. Block Matching

### Why Do This?
**Core Assumption (Non-Local Self-Similarity)**:  
Natural images exhibit the property that small local patches often have many structurally similar counterparts elsewhere in the image. For example:
- A window grille pattern repeats across the entire window
- Leaf vein textures recur within the same leaf

**Objective**:  
Collect "evidence" for signal recovery. While noise is random and uncorrelated, image structures are spatially correlated. By aggregating similar patches:
- We obtain multiple observations of the same underlying signal
- This enables separation of true signal from noise through statistical averaging

### Implementation Details
For a reference patch y_p at position p:
1. **Search Window**: Define a large neighborhood around p
2. **Similarity Measure**: Compute Euclidean distance to all other patches:
   d(p,q) = \|y_p - y_q\|^2
3. **Selection**: Keep top $K$ (e.g., 20) most similar patches
4. **Matrix Formation**: Stack selected patches as column vectors:
   Y_p = [y_1,  y_2, ..., y_K] ∈ R^(n²×K)
   (For 8×8 patches: $n²=64$, $Y_p \in \mathbb{R}^{64\times20}$)

---

## 2. Low-Rank Approximation & SVD Thresholding

### Why Low-Rank?
**Ideal Case**:  
If all similar patches were identical clean patches, their vector representations would be linearly dependent → Low-rank matrix $X_p$ (rank 1-2)

**Noise Impact**:  
Actual observations contain noise:
Y_p = X_p + Γ_p
where Γ_p represents random, full-rank noise that disrupts the low-rank structure

### Denoising via SVD
1. **Decomposition**:
   Y_p = UΣVᵀ
   - U,V: Orthogonal matrices capturing basis vectors
   - Σ: Diagonal matrix of singular values $σ_i$

2. **Thresholding Strategy**:
   - **Large Singular Values**: Retain (represent shared image structures)
   - **Small Singular Values**: Suppress (represent noise)

   **Soft-Thresholding Function**:
   S_τ(σ) = sign(σ) · max(|σ| - τ, 0)

3. **Adaptive Threshold Selection**:
   τ_p(i) = cη² / sqrt(max(σ_i(X_p)²/N(p) - η², 0) + ε)
   - η: Noise level estimate
   - N(p): Number of similar patches
   - c,ε: Regularization parameters

4. **Reconstruction**:
   X̂_p = UΣ̂Vᵀ
   where Σ̂ contains thresholded singular values

### Key Insights
- The thresholding balances noise removal and signal preservation
- Adaptive thresholds account for local noise statistics
- Final X̂_p provides an optimal low-rank estimate of the clean signal
