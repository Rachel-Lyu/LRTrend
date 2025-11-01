from tslearn.metrics import soft_dtw
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

def compute_soft_dtw_distance_matrix(data):
    distance_matrix = np.zeros((data.shape[0], data.shape[0]))
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            if i != j:
                distance_matrix[i, j] = soft_dtw(data[i], data[j])
            else:
                distance_matrix[i, j] = np.inf
    return distance_matrix

def impute_knn_soft_dtw(df, distance_matrix, k):
    smoothed_df = df.copy()
    smoothed_df.replace(0., np.nan, inplace=True)
    for i, county in enumerate(df.columns):  # Exclude the 'sum' column
        nearest_neighbors_indices = np.argsort(distance_matrix[i])[:k]
        nearest_neighbors = [df.columns[idx] for idx in nearest_neighbors_indices]
        smoothed_df[county] = df[nearest_neighbors].apply(row_smoothing, axis=1)
    return smoothed_df

def row_smoothing(row):
    non_nan_values = row.dropna()
    if len(non_nan_values) == 0:
        return 0
    non_zero_values = non_nan_values[non_nan_values != 0]
    if len(non_zero_values) == 0:
        return 0
    else:  
        return non_zero_values.mean()

date_ref = pd.read_csv(f'data/CPRadmission/CA_HRR.csv', index_col=0)
date_ref.index = pd.to_datetime(date_ref.index, format='%Y-%m-%d')
date_seq = date_ref.index
len(date_seq)
warnings.filterwarnings('ignore')
cnt_thr = 20
trim_length = 20
gt_thr = 0.

ref_names = ['CPRadmission']
ref_dfs = [pd.read_csv(f'data/{data_name}/ALL_HRR.csv', index_col=0) for data_name in ref_names]
for d_idx in range(len(ref_names)):
    ref_dfs[d_idx].index = pd.to_datetime(ref_dfs[d_idx].index, format='%Y-%m-%d')
    ref_dfs[d_idx] = ref_dfs[d_idx].loc[date_seq]
    print(ref_names[d_idx], ref_dfs[d_idx].index[0], ref_dfs[d_idx].index[-1])

noise_models = [['LogNormal'], ['LogNormal']]
penalty_values = [30, 100, 300, 1000, 3000, 10000, 30000]
zero_options = [['plus_one'], ['impute']]
correction_indicators = [False]
consensus_nulls, consensus_alts, null_intervals, alt_intervals = construct_gt(ref_dfs, date_seq, trim_length, gt_thr, cnt_thr, penalty_values, noise_models, zero_options, correction_indicators, interval_threshold = 14)

hrr_map = pd.read_csv('data/geocorr2014.csv')[['hrr', 'hrrname']].drop_duplicates().iloc[1:].sort_values(by = 'hrr')
hrr_map['hrrstate'] = hrr_map['hrrname'].apply(lambda x: x.split('-')[0].strip())
hrr2state = hrr_map[['hrr', 'hrrstate']].set_index('hrr').T.to_dict('records')[0]
state2hrr = {}
for hrr, hrrstate in hrr2state.items():
    if hrr in consensus_nulls.keys():
        if hrrstate not in state2hrr:
            state2hrr[hrrstate] = []
        state2hrr[hrrstate].append(hrr)

window_sizes = range(5, 47, 2)
# data_names = ['CPRadmission', 'JHUcase', 'CHNGclaim', 'doctorvisits', 'FBwtested', 'FBwhh', 'FBpositive', 'QUIDELpositive', 'FBwcli']
data_names = [f'GoogleS0{i}' for i in range(3, 6)]

state = 'ALL'
real_dfs = [pd.read_csv(f'data/{data_name}/{state}_HRR.csv', index_col=0) for data_name in data_names]
for d_idx in range(len(data_names)):
    real_dfs[d_idx].index = pd.to_datetime(real_dfs[d_idx].index, format='%Y-%m-%d')
    real_dfs[d_idx] = real_dfs[d_idx].loc[date_seq]
hrr_codes = [set(real_dfs[d_idx].columns) for d_idx in range(len(data_names))]
hrr_codes = set.intersection(*hrr_codes)
consensus_nulls_ = {hrr_code: consensus_nulls[hrr_code] for hrr_code in hrr_codes if hrr_code in consensus_nulls.keys()}
consensus_alts_ = {hrr_code: consensus_alts[hrr_code] for hrr_code in hrr_codes if hrr_code in consensus_alts.keys()}
alt_intervals_ = {hrr_code: alt_intervals[hrr_code] for hrr_code in hrr_codes if hrr_code in alt_intervals.keys()}

q = 0.95
cnt_thr = 0
nn_k = 3
max_delay_days = 60
outname = 'res/global_1gt.txt'
print(f'alpha = {(1-q) * 100:.2f}%')
with open(outname, 'w') as f:
    print(f'alpha = {(1-q) * 100:.2f}%', file=f)
for h_idx, hrr_df in enumerate(real_dfs):
    data_name = data_names[h_idx]
    print(data_name)
    with open(outname, 'a') as f:
        print(data_name, file=f)
    for window_size in window_sizes:
        growth_rates_df = compute_growth_rates_df(hrr_df, date_seq, window_size, regression_model = 'LogLinear', zero_option = 'impute', end_days_plus = window_size//2)
        # smoothing
        sel_df = growth_rates_df.loc[:, (growth_rates_df != 0).mean() > 0.3]
        sel_col = sel_df.columns
        # compute_soft_dtw
        distance_matrix = compute_soft_dtw_distance_matrix(sel_df.to_numpy().T)
        distance_df = pd.DataFrame(distance_matrix, index=sel_col, columns=sel_col)
        distance_df.to_csv(f'distance/{data_name}_{nn_k}NN_w{window_size}_1gt.csv')
        # compute_soft_dtw
        distance_matrix = pd.read_csv(f'distance/{data_name}_{nn_k}NN_w{window_size}_1gt.csv', index_col = 0).to_numpy()
        sel_df = impute_knn_soft_dtw(sel_df, distance_matrix, k=nn_k)
        growth_rates_df[sel_df.columns] = sel_df
        # smoothing
        hrr_codes = sorted(set(growth_rates_df.columns) & set(alt_intervals_.keys()))
        null_dist, alt_dist = construct_gt_dist(hrr_df, growth_rates_df, cnt_thr, consensus_nulls_, consensus_alts_)
        growth_threshold = np.quantile(null_dist, q)
        delay_results = []
        for hrr_code in hrr_codes:
            alt_interval = alt_intervals_[hrr_code]
            for start, end in alt_interval:
                interval_data = growth_rates_df[hrr_code][start.strftime('%y-%m-%d'):end.strftime('%y-%m-%d')]
                exceeds_threshold = interval_data[interval_data > growth_threshold]
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
        print(f'window_size = {window_size}, power = {np.mean(alt_dist > np.quantile(null_dist, q))* 100:.2f}%, delay {delay_days:.4f} days')
        with open(outname, 'a') as f:
            print(f'window_size = {window_size}, power = {np.mean(alt_dist > np.quantile(null_dist, q))* 100:.2f}%, delay {delay_days:.4f} days', file=f)