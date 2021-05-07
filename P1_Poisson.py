import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from matplotlib.tri import Triangulation
import distmesh as dm
from functions import *

# ------------------------------------------------------------------------------
# Demo script that solves the Poisson equation on a square with a circular hole.
# ------------------------------------------------------------------------------

fd = lambda p: dm.ddiff(dm.drectangle(p,-1,1,-1,1), dm.dcircle(p,0,0,0.5)) # square minus disk
# fd = lambda p: np.sqrt((p**2).sum(1))-1.0 # unit disk
print('distmeshing...')
c4n, n4e = dm.distmesh2d(fd, dm.huniform, 0.05, (-1,-1,1,1), [(-1,-1), (-1,1), (1,-1), (1,1)], fig=None)
print('done')
nE = n4e.shape[0]
nC = c4n.shape[0]
TR = Triangulation(*c4n.transpose(),n4e)
boundary_edges = freeBoundary(TR)

# Dirichlet condition on boundary of square:
bNodes = np.unique(boundary_edges)
D_indices = np.where(norm(c4n[boundary_edges[:,0],:], axis=1)>0.9)[0]
Db = boundary_edges[D_indices,:]
# Neumann condition on circle:
N_indices = list(set(range(boundary_edges.shape[0]))-set(D_indices))
Nb = boundary_edges[N_indices,:]
# Free nodes:
fNodes = np.unique(np.setdiff1d(range(nC),Db[:]))

s,m,b,vol_T,mp_T = fe_matrices(c4n, n4e, Db, Nb)

# Solve Poisson Problem:
rows, cols = np.meshgrid(fNodes,fNodes)
s_D = s[rows, cols]
u = np.zeros(nC) # Initialize solution
tu_D = np.zeros(nC)
for j in range(nC):
    tu_D[j] = D_data(*c4n[j,:])
b = b - s*tu_D
u[fNodes] = spsolve(s_D,b[fNodes]) # solve equation
u = u + tu_D

fig = plt.figure(figsize=(9,6))
ax = fig.gca(projection='3d')
ax.plot_trisurf(TR, u, cmap='viridis')
plt.show()

