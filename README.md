# ICASVD__ISMRM2025
Code and data for ISMRM 2025, Abstract 3987 (November 2024 submission)

Dependencies were installed in Summer or Fall of 2024 using pip install.
Python 3.12.7 was used.

No requirements.txt file was generated
### To Reproduce Figures:
.IMA EPI magnitude reconstruction volumes/images are located in the EPI_CMRR_170BR_magnitude folder. Use this folder path to upload the data and run the .ipynb files to reproduce figures. Install all dependencies first.

Figure 1: PCAICACombineScatter.ipynb

Figures 2 and 3: SVDICASpatialfMRI.ipynb

Figures 4 and 5: NavigatorTestPy.ipynb

### Combined ICA/SVD Method for fMRI
Given:
- X is fMRI magnitude data, voxels by time (dimension 1 by dimension 2).
- S are the spatial sources, voxels by components.
- A are the corresponding time courses, time by components.

#### Steps:
1. **ICA on X**:
   - X ~ SA' (where ' is the Hermitian operator)
2. **Zero Out Selected Components**:
   - S_0: S with r selected components (e.g., task-related) zeroed.
3. **Residual Data**:
   - X_0 = S_0 A'
4. **SVD of Residual Data**:
   - X_0 = U_0 Σ_0 V_0'
5. **Reconstruct using Components**:
   - S_structured contains the r selected components of S.
   - A_structured: Corresponding time courses (columns) in A.
Assuming the process is limited to a maximum rank of k:
- U_final = [U_0(:, 1:k-r), S_structured]
- Σ_final = [Σ_0(1:k-r, 1:k-r), 0;
             0                , 1] (for r = 1)
- V_final = [V_0(:, 1:k-r), A_structured]
##### Denoised Data:
- X_icasvd = U_final Σ_final V_final'


