import numpy as np
import pandas as pd
import cvxpy as cp
from growth import *
from smoother import *
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.special import ndtri
import warnings

def compute_growth_rates_df(hrr_df, date_seq, window_size, regression_model, zero_option, end_days_plus, seq_thr = 0):
    growth_rates = {}
    for hrr_code in hrr_df.columns:
        hrr_seq = np.array(hrr_df[hrr_code])
        if any(hrr_seq > seq_thr):
            log_growth = compute_growth(hrr_seq, window_size, regression_model, zero_option, allowable_size=None)
            growth_rates[hrr_code] = log_growth
    growth_rates_df = pd.DataFrame(growth_rates, index=date_seq[window_size - end_days_plus - 1:-end_days_plus])
    growth_rates_df.index = pd.to_datetime(growth_rates_df.index).strftime('%y-%m-%d')
    return growth_rates_df

def label_null_alt_dates(phi, gt_thr):
    alt_indices = []
    null_indices = []
    for i in range(1, len(phi)):
        if phi[i] - phi[i-1] > gt_thr:
            alt_indices.append(i)
        else:
            null_indices.append(i)
    return null_indices, alt_indices

def get_consensus_dates(date_list):
    if not date_list:
        return []
    consensus_dates = set(date_list[0])
    for dates in date_list[1:]:
        consensus_dates = consensus_dates.intersection(dates)
    return pd.to_datetime(list(consensus_dates))

def split_indices_into_intervals_threshold(indices, threshold):
    if len(indices) <= 0:
        return []
    indices = np.sort(np.unique(indices))
    differences = np.diff(indices).astype('timedelta64[D]').astype(int)
    gap_indices = [i for i, diff in enumerate(differences) if diff > threshold]
    intervals_boundaries = [0] + [i + 1 for i in gap_indices] + [len(indices)]
    intervals = [(pd.to_datetime(indices[intervals_boundaries[i]]), pd.to_datetime(indices[intervals_boundaries[i+1]-1])) for i in range(len(intervals_boundaries) - 1) if intervals_boundaries[i+1] - intervals_boundaries[i] > threshold]
    return intervals

def construct_gt(hrr_dfs, date_seq, trim_length, gt_thr, cnt_thr, penalty_values, noise_models, zero_options, correction_indicators, interval_threshold):
    for hrr_df in hrr_dfs:
        assert len(date_seq) == hrr_df.shape[0], 'len(date_seq) != hrr_df.shape[0]'
    consensus_nulls = {}
    consensus_alts = {}
    null_intervals = {}
    alt_intervals = {}

    for hrr_code in hrr_df.columns:
        skip_hrr_code = False
        cnt_seqs = []
        for hrr_df in hrr_dfs:
            cnt_seq = np.array(hrr_df[hrr_code])
            if all(cnt_seq <= cnt_thr):
                skip_hrr_code = True
            cnt_seqs.append(cnt_seq)
        if skip_hrr_code:
            continue 
        null_date_list = []
        alt_date_list = []
        for lmbda in penalty_values:
            for m_idx, noise_model in enumerate(noise_models):
                phi, _, scales, _ = multi_smoother(cnt_seqs = cnt_seqs, 
                                    date_seq = date_seq, lmbda_value = lmbda, noise_models = noise_model, 
                                    zero_options = zero_options[m_idx], correction_indicators = correction_indicators)
                        
                null_indices, alt_indices = label_null_alt_dates(phi, gt_thr)
                null_date_list.append(date_seq[null_indices])
                alt_date_list.append(date_seq[alt_indices])
                
        consensus_null = get_consensus_dates(null_date_list)
        consensus_null = consensus_null[
            (consensus_null > date_seq[trim_length]) &
            (consensus_null < date_seq[-trim_length])
        ]
        consensus_null_intervals = split_indices_into_intervals_threshold(consensus_null, interval_threshold)
        consensus_alt = get_consensus_dates(alt_date_list)
        consensus_alt = consensus_alt[
            (consensus_alt > date_seq[trim_length]) &
            (consensus_alt < date_seq[-trim_length])
        ]
        consensus_alt_intervals = split_indices_into_intervals_threshold(consensus_alt, interval_threshold)
        consensus_nulls[hrr_code] = consensus_null
        consensus_alts[hrr_code] = consensus_alt
        null_intervals[hrr_code] = consensus_null_intervals
        alt_intervals[hrr_code] = consensus_alt_intervals
    return consensus_nulls, consensus_alts, null_intervals, alt_intervals

def construct_dist(growth_rates_df, consensus_nulls, consensus_alts):
    null_dist = np.array([])
    alt_dist = np.array([])

    # Iterate over intersecting columns and known consensus date keys
    for hrr_code in np.intersect1d(growth_rates_df.columns, list(consensus_alts.keys())):
        # Processing for nulls
        null_indices = pd.to_datetime(consensus_nulls[hrr_code]).strftime('%y-%m-%d')
        null_indices = np.intersect1d(null_indices, growth_rates_df.index)
        null_values = growth_rates_df[hrr_code].loc[null_indices].to_numpy(dtype = float)
        null_values = null_values[~np.isnan(null_values)]  # Drop NaN values from null_values
        null_dist = np.concatenate((null_dist, null_values), axis=None)

        # Processing for alts
        alt_indices = pd.to_datetime(consensus_alts[hrr_code]).strftime('%y-%m-%d')
        alt_indices = np.intersect1d(alt_indices, growth_rates_df.index)
        alt_values = growth_rates_df[hrr_code].loc[alt_indices].to_numpy(dtype = float)
        alt_values = alt_values[~np.isnan(alt_values)]  # Drop NaN values from alt_values
        alt_dist = np.concatenate((alt_dist, alt_values), axis=None)

    return null_dist, alt_dist

def Stouffer(pvals, weights):
    Sstat = np.dot(weights, ndtri(pvals))
    return stats.norm.cdf(Sstat, scale = 1)

def Fisher(pvals, weights):
    Sstat = np.dot(weights, np.log(pvals))*len(pvals)
    return stats.chi2.sf(-2*Sstat, 2*len(pvals))

def Pearson(pvals, weights):
    Sstat = np.dot(weights, np.log(1-pvals))*len(pvals)
    return stats.chi2.cdf(-2*Sstat, 2*len(pvals))

def Tippett(pvals):
    Sstat = np.nanmin(pvals)
    return stats.beta.cdf(Sstat, 1, len(pvals))

def construct_gt_dist(hrr_df, growth_rates_df, cnt_thr, consensus_nulls, consensus_alts):
    hrr_mask = hrr_df.loc[pd.to_datetime(growth_rates_df.index, format='%y-%m-%d'), growth_rates_df.columns] > cnt_thr
    hrr_mask.index = pd.to_datetime(hrr_mask.index).strftime('%y-%m-%d')
    hrr_mask = hrr_mask.where(hrr_mask, other=np.nan)
    null_dist, alt_dist = construct_dist(growth_rates_df * hrr_mask, consensus_nulls, consensus_alts)
    return null_dist, alt_dist

def Simes_pvals_binary(growth_pvals_df, alpha, Simes):
    # Create an empty DataFrame to store results
    results_df = pd.DataFrame(index=growth_pvals_df.index, columns=growth_pvals_df.columns)

    if Simes:
        for index, row in growth_pvals_df.iterrows():
            sorted_P = np.sort(row)
            Simes_m = np.min(sorted_P * len(row) / np.arange(1, len(row) + 1))
            if Simes_m < alpha:
                results_df.loc[index] = row < alpha
            else: 
                results_df.loc[index] = row < -np.inf
    else:
        results_df = growth_pvals_df < alpha

    return results_df

def meta_pvals(pvals, method):
    pvals = pvals[~np.isnan(pvals)]
    pvals = pvals[pvals != 0]
    if len(pvals) == 0:
        return np.nan
    if method == 'Stouffer':
        return Stouffer(pvals, weights = np.repeat(1/len(pvals), len(pvals)))
    elif method == 'Fisher':
        return Fisher(pvals, weights = np.repeat(1/len(pvals), len(pvals)))
    elif method == 'Pearson':
        return Pearson(pvals, weights = np.repeat(1/len(pvals), len(pvals)))
    elif method == 'Tippett':
        return Tippett(pvals)