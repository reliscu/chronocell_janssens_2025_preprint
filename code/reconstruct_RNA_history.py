
import numpy as np
import scipy as sp
import scipy.linalg as la
from functools import lru_cache
from scipy.sparse.linalg import expm_multiply, eigs
from scipy.sparse import coo_matrix, csr_matrix, diags

class Reaction:
    def __init__(self, dU, dS, rate_fn):
        self.dU = dU
        self.dS = dS
        self.rate_fn = rate_fn  # function: input (# u, # s) -> rate

def define_reactions(alpha, beta, gamma):
    return [
        # Transcription: (u, s) -> (u+1, s)
        Reaction(dU=1, dS=0, rate_fn=lambda u, s: alpha),
        
        # Splicing: (u, s) -> (u-1, s+1)
        Reaction(dU=-1, dS=1, rate_fn=lambda u, s: beta * u),
        
        # Degradation: (u, s) -> (u, s-1)
        Reaction(dU=0, dS=-1, rate_fn=lambda u, s: gamma * s),
    ]

def enumerate_states(U_max, S_max):
    states = []
    index_for = {}
    for u in range(U_max + 1):
        for s in range(S_max + 1):
            idx = len(states)
            states.append((u, s))
            index_for[(u, s)] = idx
    return states, index_for

def create_transition_matrix(reactions, states, index_for, U_max, S_max):
    states, index_for = enumerate_states(U_max, S_max) 
    n_states = len(states)
    A = np.zeros((n_states, n_states))

    for i, (u, s) in enumerate(states):
        out_rate = 0.0
        
        for rxn in reactions:
            u2 = u + rxn.dU
            s2 = s + rxn.dS

            if (u2 < 0) or (u2 > U_max) or (s2 < 0) or (s2 > S_max):
                continue
                    
            rate = rxn.rate_fn(u, s)
            j = index_for[(u2, s2)]
            A[i, j] += rate
            out_rate += rate
            
        A[i, i] = -out_rate
    
    return A

def stationary_from_transition_matrix(A): 
    w, v = np.linalg.eig(A.T)
    idx = np.argmin(np.abs(w)) # Eigenvector corresponding to eigenvalue closest to 0
    pi = np.real(v[:, idx])
    if pi.sum() < 0:
        pi = -pi
    pi[pi < 0] = 0.0
    pi /= pi.sum()
    return pi

def reverse_generator(A, mu):
    ## Some refs for getting reverse time Markov generator:
    ## https://arxiv.org/abs/2502.19183 (p. 3)
    ## https://www.randomservices.org/random/markov/TimeReversal2.html
    ## https://link.springer.com/book/10.1007/978-1-4612-3038-0 (p. 239)

    # A_rev_off[i, j] = A[j, i] * mu[j] / mu[i], off-diagonal
    A_rev_off = (A.T * mu) / mu[:, None]   # uses broadcasting, no diag matrices
    np.fill_diagonal(A_rev_off, 0.0)
    A_rev = A_rev_off.copy()
    A_rev[np.diag_indices_from(A_rev)] = -A_rev_off.sum(axis=1) # Diagonals must be -sum(row)
    return A_rev

A_gene = None
X_fwd_gene = None

@lru_cache(maxsize=None)
def get_expm_per_gene(state_idx, dt):
    """
    Cached expm(A_gene[state_idx] * dt).
    """
    A_k = A_gene[state_idx]
    return la.expm(A_k * dt)

@lru_cache(maxsize=None)
def get_expm_rev_per_gene(state_idx, k, dt):
    """
    Cached expm(A_rev(state_idx, k) * dt) for BACKWARD direction.
    """
    A_rev = get_A_rev(state_idx, k)
    return la.expm(A_rev * dt)

@lru_cache(maxsize=None)
def get_A_rev(state_idx, k):
    """
    Cached reverse generator A_rev for BACKWARD direction at time index k.
    """
    mu_k = X_fwd_gene[:, k]
    return reverse_generator(A_gene[state_idx], mu_k)

def forward_distribution(A, pi, states, t, tau, state_grid):
    global A_gene, X_fwd_gene
    A_gene = A 
    
    # For each time step, calculate next state using current state
    X_fwd = np.zeros(shape=(len(states), len(t)))
    dt = np.mean(np.diff(t)) 
    # Use stationary distribution at t < 0 (system starts in steady state)
    X_fwd[:, 0] = expm_multiply(A[0].T * dt, pi)
    
    for k in range(0, len(t)-1): 
        t_curr, t_next = t[k], t[k+1]
        state_curr, state_next = state_grid[k], state_grid[k+1]
        x_curr = X_fwd[:, k]
               
        if state_curr == state_next:
            dt = t_next - t_curr
            M = get_expm_per_gene(state_curr, dt)
            x_next = x_curr @ M 
            
        else:   
            # State switch happens in current interval
            t_s = tau[state_next]

            # Split backward march into 2 steps
            dt1 = t_s - t_curr # left interval: [t_k, state_switch_time)
            A_k1 = A[state_curr]
            x_mid = expm_multiply(A_k1.T * dt1, x_curr)
      
            dt2 = t_next - t_s # right interval: [state_switch_time, t_{k+1})
            A_k2 = A[state_next]
            x_next = expm_multiply(A_k2.T * dt2, x_mid)
        
        X_fwd[:, k+1] = x_next
    
    X_fwd_gene = X_fwd
    return X_fwd
 
def backward_distribution(Y, Q, A, X_fwd, gene_idx, cell_idx, states, index_for, t, tau, state_grid):
    # Y: U and S count matrices
    # A: list of generator matrices (for gene j) for each alpha
    # X_fwd: forward distribution (for gene j)
    # Q: posterior probability of each cell (shape: # cells x len(t))
    # cell_idx: working cell index
    # gene_idx: working gene_index

    ###########################################################################

    global A_gene, X_fwd_gene
    A_gene = A
    
    # For each time step, calculate the cell's previous state using its current state

    # Initialize backwards trajectory with observed counts
    u_curr, s_curr = Y[cell_idx, gene_idx, 0], Y[cell_idx, gene_idx, 1]
    x_curr = np.zeros(shape=(A[0].shape[0],), dtype="float")
    x_curr[index_for[(u_curr, s_curr)]] = 1.0
    
    # Start backwards trajectory at cell's inferred position in time
    t_obs = np.argmax(Q[cell_idx, :])
    X_bw = np.zeros(shape=(len(states), len(t))) 
    X_bw[:, t_obs] = x_curr

    for k in reversed(range(1, t_obs + 1)):
        t_prev, t_curr = t[k-1], t[k]
        state_prev, state_curr = state_grid[k-1], state_grid[k]
        x_curr = X_bw[:, k]
        mu_k = X_fwd[:, k]
            
        if state_prev == state_curr:
            dt = t_curr - t_prev
            M_rev = get_expm_rev_per_gene(state_curr, k, dt)
            x_prev = x_curr @ M_rev
            # A_rev = reverse_generator(A[state_curr], mu_k)
            # x_prev = expm_multiply(A_rev.T * dt, x_curr) 
             
        else:
            # State switch happens in current interval             
            t_s = tau[state_curr]

            # Split backward march into 2 steps
            dt2 = t_curr - t_s # right interval: (state_switch_time, t_k]
            A_rev2 = reverse_generator(A[state_curr], mu_k)
            x_mid = expm_multiply(A_rev2.T * dt2, x_curr) 
            
            dt1 = t_s - t_prev # left interval: (t_{k-1}, state_switch_time]
            A_rev1 = reverse_generator(A[state_prev], mu_k) 
            x_prev = expm_multiply(A_rev1.T * dt1, x_mid) 
    
        X_bw[:, k-1] = x_prev
        
    return X_bw

def marginal_distribution(X, U_max, S_max, t, states):
    X_u = np.zeros(shape=(U_max + 1, len(t)))
    X_s = np.zeros(shape=(S_max + 1, len(t)))

    for k in range(0, len(t)):
        for i in range(U_max + 1):
            s_state_indices = [idx for idx, st in enumerate(states) if st[0] == i]
            # Sum probs over all possible S corresponding to the i-th U
            X_u[i, k] = np.sum(X[s_state_indices, k])
            
        for i in range(S_max + 1):
            u_state_indices = [idx for idx, st in enumerate(states) if st[1] == i]
            X_s[i, k] = np.sum(X[u_state_indices, k])
    
    return X_u, X_s

################################################################################


# def create_transition_matrix_sparse(reactions, states, index_for, U_max, S_max):
#     data = []
#     rows = []
#     cols = []

#     for i, (u, s) in enumerate(states):
#         out_rate = 0.0
#         for rxn in reactions:
#             u2 = u + rxn.dU
#             s2 = s + rxn.dS
            
#             if (u2 < 0) or (u2 > U_max) or (s2 < 0) or (s2 > S_max):
#                 continue

#             rate = rxn.rate_fn(u, s)
#             j = index_for[(u2, s2)]
#             rows.append(i)
#             cols.append(j)
#             data.append(rate)
#             out_rate += rate

#         # diagonal
#         rows.append(i)
#         cols.append(i)
#         data.append(-out_rate)

#     n_states = len(states)
#     A_sparse = csr_matrix((data, (rows, cols)), shape=(n_states, n_states))
#     return A_sparse

# def stationary_from_transition_matrix_sparse(A) 
#     AT = A.T
#     # eigenvector associated with eigenvalue closest to 0
#     w, v = eigs(AT, k=1, sigma=0.0)
#     pi = np.real(v[:, 0])
#     if pi.sum() < 0:
#         pi = -pi
#     pi[pi < 0] = 0.0
#     pi /= pi.sum()
#     return pi

# def reverse_generator_sparse(A, mu):
#     ## Some refs for getting reverse time Markov generator:
#     ## https://arxiv.org/abs/2502.19183 (p. 3)
#     ## https://www.randomservices.org/random/markov/TimeReversal2.html
#     ## https://link.springer.com/book/10.1007/978-1-4612-3038-0 (p. 239)
 
#     # Work with A^T in COO format (easy access to row/col/data)
#     A_T = A.T.tocoo()

#     # For A_T (i,j) = A[j,i]
#     # Want A_rev_off[i,j] = A[j,i] * mu[j] / mu[i] for i != j
#     data = A_T.data * mu[A_T.row] / mu[A_T.col]

#     # Remove diagonal entries (i == j)
#     mask = A_T.row != A_T.col
#     row_off = A_T.row[mask]
#     col_off = A_T.col[mask]
#     data_off = data[mask]

#     n = A.shape[0]
#     A_rev_off = coo_matrix((data_off, (row_off, col_off)), shape=(n, n)).tocsr()

#     # Row sums for diagonal (ensure rows sum to 0)
#     row_sums = np.array(A_rev_off.sum(axis=1)).ravel()

#     # A_rev = A_rev_off - diag(row_sums)
#     A_rev = A_rev_off - diags(row_sums)

#     return A_rev

@lru_cache(maxsize=None)
def get_A_rev_sparse(state_idx, k):
    mu_k = X_fwd_gene[:, k]
    return reverse_generator_sparse(A_gene[state_idx], mu_k)

def backward_distribution_sparse(Y, Q, A, X_fwd, gene_idx, cell_idx, states, index_for, t, tau, state_grid):
    # Y: U and S count matrices
    # A: list of generator matrices (for gene j) for each alpha
    # X_fwd: forward distribution (for gene j)
    # Q: posterior probability of each cell (shape: # cells x len(t))
    # cell_idx: working cell index
    # gene_idx: working gene_index

    ###########################################################################

    # For each time step, calculate the cell's previous state using its current state

    # Initialize backwards trajectory with observed counts
    u_curr, s_curr = Y[cell_idx, gene_idx, 0], Y[cell_idx, gene_idx, 1]
    x_curr = np.zeros(shape=(A[0].shape[0],), dtype="float")
    x_curr[index_for[(u_curr, s_curr)]] = 1.0
    
    # Start backwards trajectory at cell's inferred position in time
    t_obs = np.argmax(Q[cell_idx, :])
    X_bw = np.zeros(shape=(len(states), len(t))) 
    X_bw[:, t_obs] = x_curr

    for k in reversed(range(1, t_obs + 1)):
        t_prev, t_curr = t[k-1], t[k]
        state_prev, state_curr = state_grid[k-1], state_grid[k]
        x_curr = X_bw[:, k]
        mu_k = X_fwd[:, k]
            
        if state_prev == state_curr:
            dt = t_curr - t_prev
            A_rev = get_A_rev_sparse(state_curr, mu_k)
            x_prev = expm_multiply(A_rev.T * dt, x_curr) 
             
        else:
            # State switch happens in current interval             
            t_s = tau[state_curr]

            # Split backward march into 2 steps
            dt2 = t_curr - t_s # right interval: (state_switch_time, t_k]
            A_rev2 = reverse_generator_sparse(A[state_curr], mu_k)
            x_mid = expm_multiply(A_rev2.T * dt2, x_curr) 
            
            dt1 = t_s - t_prev # left interval: (t_{k-1}, state_switch_time]
            A_rev1 = reverse_generator_sparse(A[state_prev], mu_k) 
            x_prev = expm_multiply(A_rev1.T * dt1, x_mid) 
    
        X_bw[:, k-1] = x_prev
        
    return X_bw
