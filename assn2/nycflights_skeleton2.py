import pandas as pd
import numpy as np

# WARNING: Only modify code within the "WRITE YOUR CODE" blocks below.
"""
IMPORTANT NOTES
    - Edit Scope: Modify only inside WRITE YOUR CODE blocks; any change outside (including comments/spacing) may cause errors and point deductions.
    - Required Vars: Each block already includes grading-required variables; do not rename/remove them or change types/return formats.
    - Grading Rule: Evaluation is by exact match of print output—whitespace, newlines, case, and punctuation must be identical. Do not add extra prints anywhere.
    - Before Submit: Confirm no edits were made outside the designated blocks.
"""



# Task 1: Load and clean
df = pd.read_csv('nycflights.csv')  # Or local path

### WRITE YOUR CODE BELOW
df_clean = (
    df.assign(
        dep_time_num=pd.to_numeric(df['dep_time'], errors='coerce'),
        arr_time_num=pd.to_numeric(df['arr_time'], errors='coerce'),
    ).assign(
        dep_time=lambda d: d['dep_time_num'] // 100 + (d['dep_time_num'] % 100) / 60,
        arr_time=lambda d: d['arr_time_num'] // 100 + (d['arr_time_num'] % 100) / 60
    ).drop(columns=['dep_time_num', 'arr_time_num'])
    .assign(
        carrier=lambda d: d['carrier'].astype('category'),
        origin=lambda d: d['origin'].astype('category'),
    )
    .dropna(subset=['arr_delay'])
    .reset_index(drop=True)
)
### WRITE YOUR CODE ABOVE
print('Shape:', df_clean.shape)
print('Memory (MB):', df_clean.memory_usage(deep=True).sum() / 1e6)
print(df_clean.dtypes)



# Task 2: Outliers

### WRITE YOUR CODE BELOW
# TODO: IQR mask
P1 = df_clean['arr_delay'].quantile(0.25)
P3 = df_clean['arr_delay'].quantile(0.75)
IQR = P3-P1
df_filtered = df_clean[
    (df_clean['arr_delay'] < (P3 + IQR*1.5)) &
    (df_clean['arr_delay'] > (P1 - IQR*1.5))
].copy()
outliers_removed = df_clean.shape[0] - df_filtered.shape[0]
### WRITE YOUR CODE ABOVE
print('Outliers removed:', outliers_removed)

### WRITE YOUR CODE BELOW
# TODO: delay_category with pd.cut
df_filtered['delay_category'] = pd.cut(
    df_filtered['arr_delay'],
    bins=[-np.inf, 15, 60, np.inf],
    labels=['short', 'medium', 'long']
)
### WRITE YOUR CODE ABOVE
print(df_filtered['delay_category'].value_counts())



# Task 3: Groupby agg
### WRITE YOUR CODE BELOW
agg_dict = {
    'mean_arr_delay': ('arr_delay', 'mean'),
    'median_air_time': ('air_time', 'median'),
    'prop_long_delay': ('arr_delay', lambda x: (x > 60).mean()),
    'p95_distance': ('distance', lambda x: np.percentile(x, 0.95)),
}
grouped = (
    df_filtered
    .groupby((["origin","carrier"]))
    .agg(**agg_dict)
    .reset_index()
    .sort_values(by='mean_arr_delay', ascending=False)
)
### WRITE YOUR CODE ABOVE
print(grouped.head())


'''
# Task 4: Pivot and melt
### WRITE YOUR CODE BELOW
melted = (
    grouped
    .melt(metric='mean dep delay')
)
corr = 
### WRITE YOUR CODE ABOVE
print('Shape:', melted.shape, 'Corr:', round(corr, 2))




# Task 5: Windows
### WRITE YOUR CODE BELOW
worst =
top3 = 
### WRITE YOUR CODE ABOVE
print(worst)
print(top3)




# Task 6: Filter and score
### WRITE YOUR CODE BELOW
worst_dests = 
### WRITE YOUR CODE ABOVE
print(worst_dests)
'''