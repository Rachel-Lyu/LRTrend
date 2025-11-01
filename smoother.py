import numpy as np
import pandas as pd
import cvxpy as cp

def moving_average_smoother(signal, window_length = 7): 
    signal_padded = np.append(np.nan * np.ones(window_length - 1), signal)
    signal_smoothed = (np.convolve(signal_padded, np.ones(window_length, dtype=int), mode="valid")/ window_length)
    return signal_smoothed

def adjacent_imputation(cnt_seq): 
    for idx in np.where(cnt_seq <= 0)[0]: 
        if idx == 0: 
            first_valid_idx = np.min(np.where(cnt_seq > 0)[0])
            cnt_seq[:first_valid_idx] = cnt_seq[first_valid_idx]
        elif idx == len(cnt_seq) - 1: 
            cnt_seq[idx] = cnt_seq[idx - 1]
        else:
            left_idx = idx - 1
            while left_idx >= 0 and cnt_seq[left_idx] <= 0:
                left_idx -= 1
            right_idx = idx + 1
            while right_idx < len(cnt_seq) and cnt_seq[right_idx] <= 0:
                right_idx += 1
            if left_idx >= 0 and right_idx < len(cnt_seq):
                adjacent_values = [cnt_seq[left_idx], cnt_seq[right_idx]]
                cnt_seq[idx] = np.nanmean(adjacent_values)
            elif left_idx >= 0:
                cnt_seq[idx] = cnt_seq[left_idx]
            elif right_idx < len(cnt_seq):
                cnt_seq[idx] = cnt_seq[right_idx]
    return cnt_seq


def estimate_weekday_variances(log_y, log_mu_estimated, dates): 
    variances = []
    for wd in range(7): 
        mask = dates.dayofweek == wd
        variances.append(np.var(log_y[mask] - log_mu_estimated[mask]))
    return np.mean(variances)

# cnt_seq
# date_seq
# lmbda_value
# noise_model: LogNormal OR Poisson
# zero_option: remove_zeros OR impute by adjacent values OR plus_one (for noise_model == 'LogNormal' only) OR None
def weekday_correction(cnt_seq, date_seq, lmbda_value, noise_model, zero_option = None): 
    cnt_seq[cnt_seq < 0] = 0
    seq_length = len(cnt_seq)
    assert len(date_seq) == seq_length, 'len(cnt_seq) != len(date_seq)'
    assert noise_model == 'LogNormal' or noise_model == 'Poisson', 'Invalid noise_model!'
    if zero_option == 'remove_zeros': 
        valid = cnt_seq > 0
    elif zero_option == 'impute': 
        cnt_seq = adjacent_imputation(cnt_seq)
        valid = cnt_seq >= 0
    else:
        valid = cnt_seq >= 0
    
    # Construct design matrix to have weekday indicator columns and then day indicators.
    X = np.zeros((seq_length, 6 + seq_length))
    not_sunday = np.where(date_seq.dayofweek != 6)[0]
    X[not_sunday, np.array(date_seq.dayofweek)[not_sunday]] = 1
    X[np.where(date_seq.dayofweek == 6)[0], :6] = -1
    X[:, 6:] = np.eye(seq_length)
    
    b = cp.Variable(6 + seq_length)
    lmbda = cp.Parameter(nonneg=True)
    lmbda.value = lmbda_value  # Lambda value controls the smoothness
    sigma_squared = cp.Parameter(nonneg=True)
    sigma_squared.value = 0.2  # Initial guess for sigma^2
    
    # Log of expected value (log mu)
    log_mu = cp.matmul(X[valid], b)
    
    if noise_model == 'LogNormal': 
        if zero_option == 'plus_one': 
            log_y = np.log(cnt_seq[valid] + 1)
        else: 
            log_y = np.log(cnt_seq[valid])
        # Gaussian likelihood
        ll = -cp.sum(0.5 * cp.log(sigma_squared) + 0.5 / sigma_squared * cp.square(log_y - log_mu)) / np.sum(valid)
    elif noise_model == 'Poisson': 
        # Poisson likelihood
        ll = (cp.matmul(cnt_seq[valid], log_mu) - cp.sum(cp.exp(log_mu))) / np.sum(valid)
    
    # L-1 Norm of third differences, rewards smoothness
    penalty = lmbda * cp.norm(cp.diff(b[6:], 3), 1) / (seq_length - 2)
    
    # Optimization problem
    objective = cp.Minimize((-ll + penalty))
    prob = cp.Problem(objective)
    
    if noise_model == 'Poisson': 
        n_iter = 4
        for iteration in range(n_iter): 
            try: 
                prob.solve(max_iters=1000, abstol=1e-4, reltol=1e-4)
                break
            except cp.error.SolverError: 
                lmbda.value *= 2
                print(f"Poisson solver failed on iteration {iteration + 1}, doubling lambda to {lmbda.value}.")
    elif noise_model == 'LogNormal': 
        n_iter = 10
        tolerance = 1e-4
        for iteration in range(n_iter): 
            prob.solve()
            log_mu_estimated = X[valid].dot(b.value)
            new_sigma_squared_value = estimate_weekday_variances(log_y, log_mu_estimated, date_seq[valid])
            if np.abs(new_sigma_squared_value - sigma_squared.value) < tolerance: 
                break
            sigma_squared.value = new_sigma_squared_value
    
    # Correction computation
    wd_correction = np.zeros(seq_length)
    for wd in range(7): 
        mask = date_seq.dayofweek == wd
        wd_correction[mask] = cnt_seq[mask] / (np.exp(b.value[wd]) if wd < 6 else np.exp(-np.sum(b.value[:6])))
    
    # Smoothed sequence
    if zero_option == 'plus_one' and noise_model == 'LogNormal': 
        phi = np.exp(b.value[6:]) - 1
        phi[phi < 0] = 0
    else: 
        phi = np.exp(b.value[6:])
    
    alpha = np.append(b.value[:6], -np.sum(b.value[:6]))
    return alpha, sigma_squared.value, wd_correction, phi

# Construct design matrix to have weekday indicator columns and then day indicators.
def construct_design_matrix(date_seq, correction_indicators):
    n_seq = len(correction_indicators)
    n_correction = np.sum(correction_indicators)
    seq_length = len(date_seq)
    len_b = seq_length + 6 * n_correction + n_seq - 1
    X = np.zeros((seq_length * n_seq, len_b))
    not_sunday = np.where(date_seq.dayofweek != 6)[0]
    sunday = np.where(date_seq.dayofweek == 6)[0]
    i_corr = 0
    for i_seq, indicator in enumerate(correction_indicators):
        X[i_seq * seq_length:(i_seq + 1) * seq_length, :seq_length] = np.eye(seq_length)
        if indicator:
            X[i_seq * seq_length + not_sunday, i_corr * 6 + seq_length + np.array(date_seq.dayofweek)[not_sunday]] = 1
            X[i_seq * seq_length + np.where(date_seq.dayofweek == 6)[0], i_corr * 6 + seq_length:(i_corr + 1) * 6 + seq_length] = -1
            i_corr += 1
        if n_seq > 1:
            if i_seq == n_seq - 1:
                X[i_seq * seq_length:(i_seq + 1) * seq_length, seq_length + 6 * n_correction:] = -1
            else:
                X[i_seq * seq_length:(i_seq + 1) * seq_length, seq_length + 6 * n_correction + i_seq] = 1
    return X

def multi_smoother(cnt_seqs, date_seq, lmbda_value, noise_models, zero_options, correction_indicators): 
    n_seq = len(cnt_seqs)
    
    valid_indicators = []
    for c_idx, cnt_seq in enumerate(cnt_seqs):
        assert noise_models[c_idx] == 'LogNormal' or noise_models[c_idx] == 'Poisson', 'Invalid noise_models!'
        cnt_seq[cnt_seq < 0] = 0
        seq_length = len(cnt_seq)
        assert len(date_seq) == seq_length, 'len(cnt_seq) != len(date_seq)'
        if zero_options[c_idx] == 'remove_zeros': 
            valid = cnt_seq > 0
        elif zero_options[c_idx] == 'impute': 
            cnt_seq = adjacent_imputation(cnt_seq)
            valid = cnt_seq >= 0
        else:
            valid = cnt_seq >= 0
        valid_indicators.append(valid)
    valid_indicator = np.array(valid_indicators).flatten()

    X = construct_design_matrix(date_seq, correction_indicators)
    
    b = cp.Variable(X.shape[1])
    lmbda = cp.Parameter(nonneg=True)
    lmbda.value = lmbda_value  # Lambda value controls the smoothness
    sigma_squared = cp.Parameter(n_seq, nonneg=True)
    sigma_squared.value = [0.2] * n_seq  # Initial guess for sigma^2
    
    # Log of expected value (log mu)
    log_mu = cp.matmul(X[valid_indicator], b)
    
    ll_total = 0
    for c_idx, cnt_seq in enumerate(cnt_seqs):
        if noise_models[c_idx] == 'LogNormal': 
            if zero_options[c_idx] == 'plus_one': 
                log_y = np.log(cnt_seq[valid_indicators[c_idx]] + 1)
            else: 
                log_y = np.log(cnt_seq[valid_indicators[c_idx]])
            # Gaussian likelihood
            ll = -cp.sum(0.5 * cp.log(sigma_squared[c_idx]) + 0.5 / sigma_squared[c_idx] * cp.square(log_y - log_mu[c_idx * seq_length:(c_idx + 1) * seq_length])) / np.sum(valid_indicators[c_idx])
        elif noise_models[c_idx] == 'Poisson': 
            # Poisson likelihood
            ll = (cp.matmul(cnt_seq[valid_indicators[c_idx]], log_mu[c_idx * seq_length:(c_idx + 1) * seq_length]) - cp.sum(cp.exp(log_mu[c_idx * seq_length:(c_idx + 1) * seq_length]))) / np.sum(valid_indicators[c_idx])
        ll_total += ll
        
    # L-1 Norm of third differences, rewards smoothness
    penalty = lmbda * cp.norm(cp.diff(b[:seq_length], 3), 1) / (seq_length - 2)
    
    # Optimization problem
    objective = cp.Minimize((-ll_total + penalty))
    prob = cp.Problem(objective)
    
    n_iter1 = 10
    n_iter2 = 10
    tolerance = 1e-4
    for iteration in range(n_iter1): 
        try: 
            if np.all(np.array(noise_models) == 'LogNormal'):
                prob.solve()
            else:
                prob.solve(max_iters=1000, abstol=1e-4, reltol=1e-4)
            break
        except cp.error.SolverError: 
            lmbda.value *= 2
            print(f"Solver failed on iteration {iteration + 1}, doubling lambda to {lmbda.value}.")
    for iteration in range(n_iter2): 
            log_mu_estimated = X[valid_indicator].dot(b.value)
            new_sigma_squared_value = sigma_squared.value
            for c_idx, cnt_seq in enumerate(cnt_seqs):
                if noise_models[c_idx] == 'LogNormal':
                    if zero_options[c_idx] == 'plus_one': 
                        log_y = np.log(cnt_seq[valid_indicators[c_idx]] + 1)
                    else: 
                        log_y = np.log(cnt_seq[valid_indicators[c_idx]])
                    new_sigma_squared_value[c_idx] = estimate_weekday_variances(log_y, log_mu_estimated[c_idx * seq_length:(c_idx + 1) * seq_length], date_seq[valid_indicators[c_idx]])
            if np.max(np.abs(new_sigma_squared_value - sigma_squared.value)) < tolerance: 
                break
            sigma_squared.value = new_sigma_squared_value
    
    # Smoothed sequence
    if np.all(zero_options == 'plus_one') and np.all(noise_models == 'LogNormal'):
        phi = np.exp(b.value[:seq_length]) - 1
        phi[phi < 0] = 0
    else: 
        phi = np.exp(b.value[:seq_length])
    
    alphas = [np.append(b.value[i_corr * 6 + seq_length:(i_corr + 1) * 6 + seq_length], -np.sum(b.value[i_corr * 6 + seq_length:(i_corr + 1) * 6 + seq_length])) for i_corr in range(np.sum(correction_indicators))]
    if n_seq > 1:
        scales = np.append(b.value[-(n_seq - 1):], -np.sum(b.value[-(n_seq - 1):]))
    else:
        scales = np.array([0.])
    return phi, alphas, np.exp(scales), sigma_squared.value