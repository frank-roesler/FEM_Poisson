import numpy as np
from numpy.linalg import solve, det, norm
from math import factorial, pi
from scipy.sparse import coo_matrix
#------------------------------------------------------------------------------------------


def rhs(x,y):
    """Right-hand side of Poisson equation as a 2d-function of x,y"""
    val = 0

    return val

#------------------------------------------------------------------------------------------

def D_data(x,y):
    """Dirichlet boundary data as a 2d-function of x,y"""
    val = 0

    return val

#------------------------------------------------------------------------------------------

def N_data(x,y):
    """Neumann boundary data as a 2d-function of x,y"""
    val = 2*x

    return val

#------------------------------------------------------------------------------------------

def fe_matrices(c4n, n4e, Db, Nb):
    """Computes stiffness matrix s, mass matrix m and right-hand side vector b"""
    d=2 # dimension
    nNb = Nb.shape[0]
    nC,d = c4n.shape
    nE = n4e.shape[0]
    m_loc = (np.ones((d+1,d+1))+np.eye((d+1)))/((d+1)*(d+2))
    ctr = 0
    ctr_max = (d+1)**2*nE
    I = np.zeros((ctr_max+1))
    J = np.zeros((ctr_max+1))
    X_s = np.zeros((ctr_max+1))
    X_m = np.zeros((ctr_max+1))
    vol_T = np.zeros((nE))
    mp_T = np.zeros((nE, 2))
    b = np.zeros(nC)
    for j in range(nE):
        X_T = np.vstack([np.ones((1,d+1)), c4n[n4e[j,:],:].transpose() ])
        grads_T = solve(X_T, np.vstack([np.zeros((1,d)), np.eye(d)]))
        vol_T[j] = det(X_T)/factorial(d)
        mp_T[j,:]= np.sum(c4n[n4e[j,:],:], axis=0)/(d+1)
        for m in range(d+1):
            b[n4e[j, m]] = b[n4e[j, m]] + (1/(d+1))*vol_T[j]*rhs(*mp_T[j,:])
            for n in range(d+1):
                I[ctr] = n4e[j,m]
                J[ctr] = n4e[j,n]
                X_s[ctr] = vol_T[j]*grads_T[m,:]@grads_T[n,:].transpose()
                X_m[ctr] = vol_T[j]*m_loc[m,n]
                ctr += 1

    s = coo_matrix((X_s, (I, J)), shape=(nC,nC))
    m = coo_matrix((X_m, (I, J)), shape=(nC, nC))
    s = s.tocsr()
    m = m.tocsr()

    for j in range(nNb):
        vol_S = norm(c4n[Nb[j,0],:] - c4n[Nb[j,1],:])
        mp_S = np.sum(c4n[Nb[j,:],:], axis=0) / d
        for k in range(d):
            b[Nb[j,k]] = b[Nb[j,k]] + (1/d) * vol_S * N_data(*mp_S)


    return s,m,b,vol_T,mp_T

#------------------------------------------------------------------------------------------

def freeBoundary(TR):
    """computes boundary edges of triangulation TR"""
    n4e = TR.triangles
    neighbors = TR.neighbors
    boundary_nodes0 = np.where(neighbors==-1)
    boundary_nodes1 = (boundary_nodes0[0], (boundary_nodes0[1]+1)%3)
    bnodes0 = n4e[boundary_nodes0]
    bnodes1 = n4e[boundary_nodes1]

    return np.array([bnodes0,bnodes1]).transpose()



