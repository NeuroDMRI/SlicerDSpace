<div align="center">

# Diffusion Gradient Frame, 22 November 2026

[Project](../../README.md) | [Documentation](../README.md) | [中文](GRADIENT_FRAME_20261122.zh-CN.md)

</div>

---

This note says which results change after the gradient-frame correction.

## Rule

FSL `.bvec` values are given in the voxel axes of the NIfTI image. The world
direction of a gradient is `R F b`. `R` is the image affine with unit columns.
`F` flips the first axis when the affine determinant is positive. This is the
FSL and MRtrix convention. Before this change, some routes wrote or read the
gradients in a different frame.

## Inputs that do not change

- A NIfTI image stored with a negative determinant and axis-aligned axes (for
  example LAS, the usual FSL and HCP storage). UKF, the native tensor fit,
  the standalone DDSeg DTI features, DIPY tracking, DKI and NODDI give the same
  results as before. The converted UKF and tensor headers are byte-identical.
- The tensor-fit matrix on an axis-aligned grid or on isotropic voxels: it is
  bit-identical. (A positive determinant still changes the gradients, as below.)
- An imported NRRD whose DWIConvert `.bvec` already matches the source
  gradients: the `.bvec` file is kept byte-identical.

## Inputs that change

| Input | Routes that change |
|---|---|
| Positive determinant (for example RAS storage) | UKF, native tensor, standalone DDSeg DTI features, DIPY tracking, DKI, NODDI |
| Oblique or permuted grid, negative determinant | UKF and standalone DDSeg DTI features (DWIConvert dropped the rotation) |
| Oblique grid with anisotropic voxels | Native tensor fit: tensor direction, FA, MD and eigenvalues |
| Oblique NRRD import | The imported `.bvec` file is rewritten from the source gradients |

Fractional anisotropy, mean diffusivity and the eigenvalues do not depend on
the gradient frame. They change only in the oblique anisotropic tensor-fit case.
Tensor directions, colour maps and streamlines change in every listed case.

## Cached results

The native DTI, diffusion header and registered DTI caches record the code of
the converter, the tensor fit and the preparation step. For a NIfTI image whose
axes are exactly aligned with the scanner axes (in any order) and whose
determinant is negative, these records keep their earlier values, so finished
results are reused. Every other input is computed again once. Stages whose cache
records the native DTI cache (for example the registered DTI bundle and the
microstructure maps) are then computed again too.
UKF, DIPY tracking, DKI, NODDI and NRRD import are computed again only for the
inputs listed above. The standalone DDSeg DTI feature folder has no cache key:
delete `DTI/` in its output folder to compute it again.
