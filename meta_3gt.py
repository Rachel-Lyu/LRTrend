import numpy as np
import pandas as pd
import cvxpy as cp
from growth import *
from utils import *
from smoother import *
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.special import ndtri
import warnings

date_ref = pd.read_csv(f'data/CPRadmission/CA_HRR.csv', index_col=0)
date_ref.index = pd.to_datetime(date_ref.index, format='%Y-%m-%d')
date_seq = date_ref.index
len(date_seq)
warnings.filterwarnings('ignore')
cnt_thr = 20
trim_length = 20
gt_thr = 0.

ref_names = ['CPRadmission', 'JHUcase', 'CHNGclaim']
ref_dfs = [pd.read_csv(f'data/{data_name}/ALL_HRR.csv', index_col=0) for data_name in ref_names]
for d_idx in range(len(ref_names)):
    ref_dfs[d_idx].index = pd.to_datetime(ref_dfs[d_idx].index, format='%Y-%m-%d')
    ref_dfs[d_idx] = ref_dfs[d_idx].loc[date_seq]
    print(ref_names[d_idx], ref_dfs[d_idx].index[0], ref_dfs[d_idx].index[-1])

noise_models = [['LogNormal', 'LogNormal', 'Poisson'], ['LogNormal', 'LogNormal', 'Poisson']]
penalty_values = [30, 100, 300, 1000, 3000, 10000, 30000]
zero_options = [['plus_one', 'plus_one', None], ['impute', 'impute', None]]
correction_indicators = [False, True, True]
consensus_nulls, consensus_alts, null_intervals, alt_intervals = construct_gt(ref_dfs, date_seq, trim_length, gt_thr, cnt_thr, penalty_values, noise_models, zero_options, correction_indicators, interval_threshold = 14)

# window_sizes = [11, 21, 31]
window_sizes = range(5, 47, 2)
data_names = ['CPRadmission', 'JHUcase', 'CHNGclaim', 'doctorvisits', 'FBwtested', 'FBwhh', 'FBpositive', 'QUIDELpositive', 'FBwcli'] + [f'GoogleS0{i}' for i in range(3, 6)]
meta_method = 'Stouffer'
meta_lists = [['CPRadmission', 'JHUcase', 'CHNGclaim', 'doctorvisits', 'FBwtested', 'FBwhh', 'FBpositive', 'QUIDELpositive', 'FBwcli'] + [f'GoogleS0{i}' for i in range(3, 6)],
              ['CPRadmission', 'JHUcase', 'CHNGclaim', 'doctorvisits', 'FBwtested', 'FBwhh', 'FBpositive', 'QUIDELpositive'] + [f'GoogleS0{i}' for i in range(3, 6)],
              ['doctorvisits', 'FBwtested', 'FBwhh', 'FBpositive', 'QUIDELpositive', 'FBwcli'] + [f'GoogleS0{i}' for i in range(3, 6)],
              ['doctorvisits', 'FBwtested', 'FBwhh', 'FBpositive', 'QUIDELpositive'] + [f'GoogleS0{i}' for i in range(3, 6)],
              ['CPRadmission', 'JHUcase', 'CHNGclaim']]

state = 'ALL'
real_dfs = [pd.read_csv(f'data/{data_name}/{state}_HRR.csv', index_col=0) for data_name in data_names]
for d_idx in range(len(data_names)):
    real_dfs[d_idx].index = pd.to_datetime(real_dfs[d_idx].index, format='%Y-%m-%d')
    real_dfs[d_idx] = real_dfs[d_idx].loc[date_seq]
    print(data_names[d_idx], real_dfs[d_idx].index[0], real_dfs[d_idx].index[-1])
hrr_codes = [set(real_dfs[d_idx].columns) for d_idx in range(len(data_names))]
hrr_codes = set.intersection(*hrr_codes)
consensus_nulls_ = {hrr_code: consensus_nulls[hrr_code] for hrr_code in hrr_codes if hrr_code in consensus_nulls.keys()}
consensus_alts_ = {hrr_code: consensus_alts[hrr_code] for hrr_code in hrr_codes if hrr_code in consensus_alts.keys()}
alt_intervals_ = {hrr_code: alt_intervals[hrr_code] for hrr_code in hrr_codes if hrr_code in alt_intervals.keys()}
hrr_codes = list(sorted(consensus_nulls_.keys()))

q = 0.95
cnt_thr = 0
max_delay_days = 60
outname = 'res/meta_full_3gt.txt'
print(f'alpha = {(1-q) * 100:.2f}%')
with open(outname, 'w') as f:
    print(f'alpha = {(1-q) * 100:.2f}%', file=f)
for window_size in window_sizes:
    print(f"Window size {window_size}")
    with open(outname, 'a') as f:
        print(f"Window size {window_size}", file=f)
    growth_rates_dfs = {}
    growth_pvals_dfs = {}
    for h_idx, hrr_df in enumerate(real_dfs):
        growth_rates_df = compute_growth_rates_df(hrr_df, date_seq, window_size, regression_model = 'LogLinear', zero_option = 'impute', end_days_plus = window_size//2, seq_thr = 0)
        null_dist, alt_dist = construct_gt_dist(hrr_df, growth_rates_df, cnt_thr, consensus_nulls_, consensus_alts_)
        growth_pvals_df = pd.DataFrame(index=growth_rates_df.index, columns=hrr_codes)
        for column in hrr_codes:
            for index in growth_rates_df.index:
                try:
                    growth_pvals_df.at[index, column] = np.mean(null_dist > growth_rates_df.at[index, column])
                except KeyError:
                    growth_pvals_df.at[index, column] = np.nan
        growth_rates_dfs[data_names[h_idx]] = growth_rates_df.copy()
        growth_pvals_dfs[data_names[h_idx]] = growth_pvals_df.copy()
        p_null_dist, p_alt_dist = construct_dist(growth_pvals_df, consensus_nulls_, consensus_alts_)
        pvals_threshold = np.quantile(p_null_dist, 1-q)
        delay_results = []
        for key, alt_interval in alt_intervals_.items():
            for start, end in alt_interval:
                interval_data = growth_pvals_df[key][start.strftime('%y-%m-%d'):end.strftime('%y-%m-%d')]
                exceeds_threshold = interval_data[interval_data < pvals_threshold]
                if not exceeds_threshold.empty:
                    first_exceedance_time = pd.to_datetime(exceeds_threshold.index[0], format='%y-%m-%d')
                    delay = (first_exceedance_time - start).days
                    delay_results.append(delay)
                else:
                    delay_results.append(max_delay_days)
        if len(delay_results) == 0:
            delay_days = max_delay_days
        else:
            delay_days = np.mean(delay_results)
        print(f'{data_names[h_idx]}, power = {np.mean(p_alt_dist < np.quantile(p_null_dist, 1-q))* 100:.2f}%, pthr = {pvals_threshold:.4f}, delay {delay_days:.4f} days')
        with open(outname, 'a') as f:
            print(f'{data_names[h_idx]}, power = {np.mean(p_alt_dist < np.quantile(p_null_dist, 1-q))* 100:.2f}%, pthr = {pvals_threshold:.4f}, delay {delay_days:.4f} days', file=f)

    for m_idx, meta_list in enumerate(meta_lists): 
        growth_pvals_df = pd.DataFrame(index=growth_rates_df.index, columns=hrr_codes)
        for column in hrr_codes:
            for index in growth_rates_df.index:
                pvals = np.array([growth_pvals_dfs[df_name].at[index, column] for df_name in meta_list])
                growth_pvals_df.at[index, column] = meta_pvals(pvals, meta_method)
        p_null_dist, p_alt_dist = construct_dist(growth_pvals_df, consensus_nulls_, consensus_alts_)
        pvals_threshold = np.quantile(p_null_dist, 1-q)
        delay_results = []
        for key, alt_interval in alt_intervals_.items():
            for start, end in alt_interval:
                interval_data = growth_pvals_df[key][start.strftime('%y-%m-%d'):end.strftime('%y-%m-%d')]
                exceeds_threshold = interval_data[interval_data < pvals_threshold]
                if not exceeds_threshold.empty:
                    first_exceedance_time = pd.to_datetime(exceeds_threshold.index[0], format='%y-%m-%d')
                    delay = (first_exceedance_time - start).days
                    delay_results.append(delay)
                else:
                    delay_results.append(max_delay_days)
        if len(delay_results) == 0:
            delay_days = max_delay_days
        else:
            delay_days = np.mean(delay_results)
        print(f'{m_idx}_{meta_method}, power = {np.mean(p_alt_dist < np.quantile(p_null_dist, 1-q))* 100:.2f}%, pthr = {pvals_threshold:.4f}, delay {delay_days:.4f} days')
        with open(outname, 'a') as f:
            print(f'{m_idx}_{meta_method}, power = {np.mean(p_alt_dist < np.quantile(p_null_dist, 1-q))* 100:.2f}%, pthr = {pvals_threshold:.4f}, delay {delay_days:.4f} days', file=f)

    for m_idx, meta_list in enumerate(meta_lists): 
        growth_cnts_df = pd.DataFrame(index=growth_rates_df.index, columns=hrr_codes)
        for column in hrr_codes:
            for i, index in enumerate(growth_rates_df.index):
                # growths = [growth_rates_dfs[df_name][column][i] for df_name in meta_list]
                growths = []
                for df_name in meta_list:
                    try:
                        growths.append(growth_rates_dfs[df_name][column][i])
                    except KeyError:
                        growths.append(0)
                growth_cnts_df.at[index, column] = np.sum((np.array(growths)>0))
        c_null_dist, c_alt_dist = construct_dist(growth_cnts_df, consensus_nulls_, consensus_alts_)
        cnts_threshold = np.quantile(c_null_dist, q)
        delay_results = []
        for key, alt_interval in alt_intervals_.items():
            for start, end in alt_interval:
                interval_data = growth_cnts_df[key][start.strftime('%y-%m-%d'):end.strftime('%y-%m-%d')]
                exceeds_threshold = interval_data[interval_data > cnts_threshold]
                if not exceeds_threshold.empty:
                    first_exceedance_time = pd.to_datetime(exceeds_threshold.index[0], format='%y-%m-%d')
                    delay = (first_exceedance_time - start).days
                    delay_results.append(delay)
                else:
                    delay_results.append(max_delay_days)
        if len(delay_results) == 0:
            delay_days = max_delay_days
        else:
            delay_days = np.mean(delay_results)
        print(f'{m_idx}_Stolerman, power = {np.mean(c_alt_dist > np.quantile(c_null_dist, q))* 100:.2f}%, cnt_thr = {cnts_threshold:.2f}, delay {delay_days:.4f} days')
        with open(outname, 'a') as f:
            print(f'{m_idx}_Stolerman, power = {np.mean(c_alt_dist > np.quantile(c_null_dist, q))* 100:.2f}%, cnt_thr = {cnts_threshold:.2f}, delay {delay_days:.4f} days', file=f)