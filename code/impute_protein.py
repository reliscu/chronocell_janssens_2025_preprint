import numpy as np

def impute_protein(X_bw, Q, t, deg_rate, transl_rate=1):
    Q_max_idx = np.argmax(Q, axis=1) # Each cell's time along the trajectory

    y0 = X_bw[:, 0] # Steady-state RNA abundance per gene
    ss_rate = transl_rate / deg_rate # Steady-state protein production rate 
    p0 = ss_rate * y0 # Steady-state protein abundance
    p_old = p0 * np.exp(-deg_rate * t[Q_max_idx]) # Pre-existing protein that has not yet degraded at the time each cell was observed

    t_reshape = t.reshape((-1, 1))
    t_diff = t_reshape.T - t_reshape
    t_diff = t_diff[:, Q_max_idx] # Each column corresponds to a cell: contains time steps with cumulative duration of time leading up to its observed time

    decay_matrix = np.exp(-t_diff * deg_rate) # Decay_matrix[i, m] = decay factor for protein made from RNA available at t_i in cell m
    mask = (t_diff >= 0) # Protein abundance at time t_m can't come from RNA at time t_i > t_m (cell's observed time)
    mask = np.broadcast_to(mask, decay_matrix.shape)
    decay_matrix = np.where(mask, decay_matrix, 0) 
    
    dt = np.diff(t, prepend=-t[1])
    X_bw_dt = X_bw * dt # Multiply each timepoint's RNA by its corresponding time step size (Riemann approximation) 
    X_bw_dt[X_bw_dt < 0] = 0 # X_bw was populated with '-1' for time points after the cell was observed

    p_new = (decay_matrix * X_bw_dt.T).sum(axis=0) # Integrate RNA counts still surviving up to each time point (until the cell was observed)
    P = p_old + p_new # Protein abundance in each cell = pre-existing protein + newly synthesized protein
           
    return P