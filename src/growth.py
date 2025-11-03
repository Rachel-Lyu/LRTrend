from smoother import adjacent_imputation
import numpy as np
import statsmodels.api as sm
from scipy.stats import norm, t

# zero_option: remove_zeros OR impute by adjacent values OR plus_one (for noise_model == 'LogNormal' only)
def compute_growth(cnt_seq, window_size, regression_model, zero_option, allowable_size = None): 
    cnt_seq[cnt_seq < 0] = 0
    seq_length = len(cnt_seq)
    if np.all(cnt_seq == 0): 
        return np.zeros(seq_length - window_size + 1)
    elif np.any(cnt_seq == 0): 
        if zero_option == 'remove_zeros' and allowable_size is not None: 
            valid_indices = np.where(cnt_seq > 0)[0]
            allowable_size += window_size
            first_derivatives = []
            for i in range(seq_length - window_size + 1): 
                i += window_size // 2
                sorted_indices = valid_indices[np.argsort(np.abs(valid_indices - i))]
                closest_indices = sorted_indices[:window_size]
                if np.any(np.abs(closest_indices - i) > allowable_size // 2): 
                    first_derivative = 0
                else: 
                    window = cnt_seq[closest_indices]
                    x_values = closest_indices - min(closest_indices)
                    first_derivative = local_regression(x_values, window, regression_model)[0]
                first_derivatives.append(first_derivative)
            return np.array(first_derivatives)
        elif zero_option == 'impute': 
            cnt_seq = adjacent_imputation(cnt_seq)
        elif zero_option == 'plus_one': 
            cnt_seq = cnt_seq + 1
    
    windows = [cnt_seq[i:i+window_size] for i in range(seq_length - window_size + 1)]
    x_values = np.arange(window_size)
    first_derivatives = []
    for window in windows: 
        first_derivative = local_regression(x_values, window, regression_model)[0]
        first_derivatives.append(first_derivative)
    return np.array(first_derivatives)

def local_regression(x, y, regression_model, beta_threshold=0): 
    if np.all(y == y[0]): 
        return 0, 0, np.nan, np.nan
    if np.any(np.convolve(y == 0, [1, 1], mode='valid') == 2):
        regression_model = 'LogLinear'
    else:
        y_non_zero = y[y != 0]
        if len(y_non_zero) > 1 and np.allclose(y_non_zero[1:] / y_non_zero[:-1], y_non_zero[1] / y_non_zero[0], atol=1e-5):
            regression_model = 'LogLinear'
    try: 
        if regression_model == 'LogLinear': 
            y_filtered = y[y > 0]
            if len(y_filtered) <= 2:
                return 0, 0, np.nan, np.nan
            x_filtered = x[y > 0]
            model_linear = sm.OLS(np.log(y_filtered), sm.add_constant(x_filtered)).fit()
            beta = model_linear.params[1]
            se = model_linear.bse[1]
            if np.abs(se) <= 1e-5:
                return beta, se, np.nan, np.nan
            # Calculate the t-statistic
            statistic = (beta - beta_threshold) / se
            p_value = 1 - t.cdf(statistic, df=len(x) - 2)
        else: 
            model_poisson = sm.GLM(y, sm.add_constant(x), family=sm.families.Poisson()).fit()
            if regression_model == 'Poisson': 
                beta = model_poisson.params[1]
                se = model_poisson.bse[1]
            elif regression_model == 'NegativeBinomial': 
                mu = model_poisson.mu
                # variance = (y - mu) ** 2
                # alpha = ((variance - y) / mu).mean()
                # model_neg_bin = sm.GLM(y, sm.add_constant(x), family=sm.families.NegativeBinomial(alpha=(variance.mean() - mu.mean()) / (mu.mean()**2))).fit()
                alpha = (np.var((y - mu), ddof=1) - mu.mean()) / (mu**2).mean()
                model_neg_bin = sm.GLM(y, sm.add_constant(x), family=sm.families.NegativeBinomial(alpha=alpha)).fit()
                beta = model_neg_bin.params[1]
                se = model_neg_bin.bse[1]
            # Calculate the z-statistic
            statistic = (beta - beta_threshold) / se
            p_value = 1 - norm.cdf(statistic)
        return beta, se, statistic, p_value
    
    except Exception as e: 
        print("An error occurred: ", e)
        print("x values: ", x)
        print("y values: ", y)
        return None
