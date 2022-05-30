# FEM Poisson

This collection of Python codes provides a simple, completely transparent implementation of the P1 finite element method to solve the Poisson equation on a 2d domain of arbitrary shape. The code is a translation of the Matlab code provided in [Bartels, Sören. Numerical approximation of partial differential equations. Vol. 64. Springer, 2016](https://aam.uni-freiburg.de/agba/prof/books.html). The meshing in the demo scripts depends on the [PyDistmesh](https://pypi.org/project/PyDistMesh/) package. Other dependencies:
* [matplotlib](https://matplotlib.org/)
* [numpy](https://numpy.org/)
* [scipy](https://www.scipy.org/)
#### Functions:
* `freeBoundary(TR)`: computes boundary edges from triangulation object TR.
* `fe_matrices(c4n, n4e, Db, Nb)`: computes stiffness and mass matrix and right-hand side vector from triangulation and data of the PDE.
* `rhs(x,y)`: Right-hand side of Poisson equation as a 2d-function of `x,y`.
* `D_data(x,y)`: Dirichlet boundary data as a 2d-function of `x,y`.
* `N_data(x,y)`: Neumann boundary data as a 2d-function of `x,y`.
#### Demo scripts:
* `D_Eigenfunctions.py`: Computes the Dirichlet eigenfunctions of the Laplacian on the unit disk.
* `P1_Poisson.py`: Solves the Poisson equation on a square with a circular hole.

Any comments or queries are welcome at https://frank-roesler.github.io/contact/
