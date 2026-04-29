import numpy as np

def impute_protein(X_bw, Q, t, deg_rate, transl_rate=1):
    Q_max_idx = np.argmax(Q, axis=1) # Each cell's time along the trajectory

    # Pre-existing protein
    y0 = X_bw[:, 0] # Steady-state RNA abundance (RNA at start of first interval)
    ss_rate = transl_rate / deg_rate # Definition of steady-state protein production rate 
    p0 = ss_rate * y0 # Protein at start of first interval 
    p_old = p0 * np.exp(-deg_rate * t[Q_max_idx]) # Protein from start of first interval that survived per cell

    dt = np.concatenate([np.diff(t), [0.0]])
    X_bw_clean = np.where(X_bw < 0, 0, X_bw) # clean -1 placeholders (X_bw was populated with '-1' for time points after the cell was observed)
    X_bw_dt = X_bw_clean * dt # Multiply each timepoint's RNA by its corresponding time step size (Riemann approximation) 

    # Protein produced over start of trajectory to the time a cell was sampled
    t_reshape = t.reshape((-1, 1))
    t_diff = t_reshape.T - t_reshape
    t_diff = t_diff[:, Q_max_idx] # Each column contains time steps up to cell's observed time
    mask = (t_diff > 0)  
    decay_matrix = np.exp(-np.maximum(t_diff, 0) * deg_rate)
    decay_matrix = np.where(mask, decay_matrix, 0) # decay_matrix[i, m]: decay factor for protein at t[m] from RNA available at t[i] in cell m 
    
    p_new = (decay_matrix * X_bw_dt.T).sum(axis=0) # Integrate protein counts from each interval
    P = p_old + p_new # Protein abundance in each cell = pre-existing protein + newly synthesized protein
           
    return P