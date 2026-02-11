> Under Construction

# Solving Forward and Inverse PDEs on Unknown Manifolds with Physics-Informed Neural Operators

The data and code for the paper [Solving Forward and Inverse PDE Problems on Unknown Manifolds via Physics-Informed Neural Operators](https://epubs.siam.org/doi/full/10.1137/24M1675254), *SIAM Journal on Scientific Computing*, 48 (1), C136–C163, 2026.

## Data

The datasets are generated from the MATLAB code in the [data](data) folder. The repository includes three approaches for approximating differential operators on unknown manifolds:
- [Diffusion Maps](data/DM/)
- [Radial Basis Functions](data/RBF/)
- [Generalized Moving Least Squares](data/GMLS/)

The full datasets are available on [OneDrive](https://yaleedu-my.sharepoint.com/:f:/g/personal/lu_lu_yale_edu/IgAorv2ASH9jR6OPj1BBIrAJAeUlNN0Lpc-KXM5iEJdXxq0?e=CbRCyk).

## Code

The code for solving the forward and inverse problems on unknown manifolds can be found in the [src](src) folder. 
- [Second-order linear elliptic PDE on torus](src/linear_torus/)
- [Second-order linear elliptic PDE on semi-torus with Dirichlet boundary conditions](src/semi_torus/)
- [Nonlinear PDE on torus](src/nonlinear_torus/)
- [Second-order linear elliptic PDE on Bunny](src/bunny)
- [Application to solving Bayesian inverse problems](src/inverse_problem)

For physics-informed training with observational data, consistency between the data and the governing PDE is essential. We therefore provide a validation procedure to check whether the generated data is consistent with the PDE (see [example](src/bunny/check_data_pytorch.py)).


## Cite this work

If you use this data or code for academic research, you are encouraged to cite the following paper:

```
@article{Jiao2026PINOManifold,
  author  = {Jiao, Anran and Yan, Qile and Harlim, John and Lu, Lu},
  title   = {Solving Forward and Inverse PDE Problems on Unknown Manifolds via Physics-Informed Neural Operators},
  journal = {SIAM Journal on Scientific Computing},
  volume  = {48},
  number  = {1},
  pages   = {C136--C163},
  year    = {2026},
  doi     = {https://doi.org/10.1137/24M1675254}
}
```

## Questions

To get help on how to use the data or code, simply open an issue in the GitHub "Issues" section.
