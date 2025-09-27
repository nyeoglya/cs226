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
        dep_hour=lambda d: (d['dep_time_num'] // 100) + (d['dep_time_num'] % 100) / 60,
        arr_hour=lambda d: (d['arr_time_num'] // 100) + (d['arr_time_num'] % 100) / 60,
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
Q1 = df_clean['arr_delay'].quantile(0.25)
Q3 = df_clean['arr_delay'].quantile(0.75)
IQR = Q3 - Q1
df_filtered = df_clean.loc[(df_clean['arr_delay'] >= Q1-1.5*IQR) & (df_clean['arr_delay'] <= Q3+1.5*IQR)].copy()
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
    'p95_distance': ('distance', lambda x: np.percentile(x, 95)),
}
grouped = (
    df_filtered
    .groupby(['origin', 'carrier'])
    .agg(**agg_dict)
    .reset_index()
    .sort_values('mean_arr_delay', ascending=False)
)
### WRITE YOUR CODE ABOVE
print(grouped.head())



# Task 4: Pivot and melt
### WRITE YOUR CODE BELOW
top3_carriers = df_filtered['carrier'].value_counts().head(3).index
melted = (
    df_filtered
    .pivot_table(index='month', columns='carrier', values='dep_delay', aggfunc='mean', fill_value=0, margins=True, observed=True)
    .reset_index()
    .melt(id_vars='month', var_name='carrier', value_name='mean_dep_delay')
    .loc[lambda d: (d['month'] != 'All') & (d['carrier'] != 'All')]
    .loc[lambda d: d['carrier'].isin(top3_carriers)]
    .assign(metric='mean_dep_delay')
)
corr = (
    melted
    .assign(month=pd.to_numeric(melted['month'], errors='coerce'))
    .dropna(subset=['month', 'mean_dep_delay'])[['month', 'mean_dep_delay']]
    .corr()
    .iloc[0, 1]
)
### WRITE YOUR CODE ABOVE
print('Shape:', melted.shape, 'Corr:', round(corr, 2))

print(df_filtered
    .pivot_table(index='month', columns='carrier', values='dep_delay', aggfunc='mean', fill_value=0, margins=True, observed=True)
    .reset_index()
    .melt(id_vars='month', var_name='carrier', value_name='mean_dep_delay'))


# Task 5: Windows
### WRITE YOUR CODE BELOW
worst = (
    df_filtered
    .groupby(['month','carrier'], observed=True)['arr_delay'].mean().reset_index()
    .assign(rank=lambda d: d.groupby('month')['arr_delay'].rank(method='dense', ascending=True))
    .loc[lambda d: d.groupby('month')['rank'].transform('max') == d['rank']]
    .set_index('month')['carrier']
    .sort_index()
)
top3 = (
    df_filtered
    .groupby(['carrier','month'], observed=True)['arr_delay'].mean().reset_index()
    .sort_values(['carrier','month'])
    .assign(cum_avg=lambda d: d.groupby('carrier', observed=True)['arr_delay'].expanding().mean().reset_index(level=0, drop=True))
    .groupby('carrier', observed=True)['cum_avg'].last().reset_index()
    .assign(rank=lambda d: d['cum_avg'].rank(method='dense', ascending=False))
    .loc[lambda d: d['rank'] <= 3]
    .set_index('carrier')['cum_avg']
    .sort_values(ascending=False)
)
### WRITE YOUR CODE ABOVE
print(worst)
print(top3)




# Task 6: Filter and score
### WRITE YOUR CODE BELOW
worst_dests = (
    df_filtered
    .loc[lambda d: (d['distance'] > 1000) & (d['air_time'] > d['air_time'].quantile(0.90))]
    .assign(score=lambda d: (d['dep_delay'] + d['arr_delay']) / d['distance'] * 100)
    .groupby('dest', observed=True)
    .agg(mean_score=('score','mean'), count=('score','size'))
    .reset_index()
    .loc[lambda d: d['count'] > 1000]
    .sort_values('mean_score', ascending=False)
    .head(5)
)
### WRITE YOUR CODE ABOVE
print(worst_dests)
