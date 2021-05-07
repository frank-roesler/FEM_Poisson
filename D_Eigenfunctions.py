from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import distmesh as dm
from functions import *

# ---------------------------------------------------------------------------------------
# Demo script that computes the first 24 eigenfunctions on a square with a circular hole.
# ---------------------------------------------------------------------------------------

# fd = lambda p: dm.ddiff(dm.drectangle(p,-1,1,-1,1), dm.dcircle(p,0,0,0.5)) # square minus disk
fd = lambda p: np.sqrt((p**2).sum(1))-1.0 # unit disk
print('distmeshing...')
c4n, n4e = dm.distmesh2d(fd, dm.huniform, 0.05, (-1,-1,1,1), [(-1,-1), (-1,1), (1,-1), (1,1)], fig=None)
print('done')
nE = n4e.shape[0]
nC = c4n.shape[0]
TR = Triangulation(*c4n.transpose(),n4e)
boundary_edges = freeBoundary(TR)

Db = np.unique(boundary_edges) # Dirichlet condition everywhere
Nb = np.array([])
s,m,b,vol_T,mp_T = fe_matrices(c4n, n4e, Db, Nb)
s = s.tocsr()
m = m.tocsr()

# Compute Dirichlet Eigenfunctions:
fNodes = np.unique(np.setdiff1d(range(nC),Db))
rows, cols = np.meshgrid(fNodes,fNodes)
s_D = s[rows, cols]
m_D = m[rows, cols]
N_eig = 24
vals, vecs = eigs(s_D, k=N_eig, M=m_D, which='SM')
u = np.zeros(nC)

fig = plt.figure(figsize=(12,8))
for i in range(N_eig):
    u[fNodes] = vecs[:, i]
    u = u/np.sqrt((u.T@(m@u)))
    ax = fig.add_subplot(4,6,i+1)
    tcf = ax.tricontourf(TR, u, 50, cmap='viridis', vmin=-1.3, vmax=1.3)
    ax.set_title('EF number {}'.format(i+1))
    # fig.colorbar(tcf)

plt.tight_layout()
plt.show()




