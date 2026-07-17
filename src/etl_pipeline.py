import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)


print("Building dim_tower...")
df_404 = pd.read_csv('data/raw/mobile_network_coverage_india/404.csv')
df_405 = pd.read_csv('data/raw/mobile_network_coverage_india/405.csv')
towers = pd.concat([df_404, df_405], ignore_index=True)

# Drop useless columns
towers = towers[['radio', 'mcc', 'mnc', 'lac', 'cid', 'long', 'lat', 'range']]
towers = towers.rename(columns={'long': 'longitude', 'lat': 'latitude'})
towers['tower_id'] = towers.index + 1  

# Merge operator + circle info
operators = pd.read_csv('data/raw/mobile_network_coverage_india/MCC-MNC India.csv')
towers = towers.merge(operators, on=['mcc', 'mnc'], how='left')
towers['operator'] = towers['operator'].fillna('Unknown')
towers['circle'] = towers['circle'].fillna('Unknown')

print(f"dim_tower shape: {towers.shape}")
towers.to_csv('data/processed/dim_tower.csv', index=False)


dim_operator = towers[['operator', 'circle']].drop_duplicates().reset_index(drop=True)
dim_operator['operator_id'] = dim_operator.index + 1
dim_operator.to_csv('data/processed/dim_operator.csv', index=False)
print(f"dim_operator shape: {dim_operator.shape}")


print("\nBuilding fact_rf_readings...")
fact = pd.read_csv('data/raw/cellular_network_performance/cellular_performance.csv')
fact['Timestamp'] = pd.to_datetime(fact['Timestamp'], format='mixed', dayfirst=False, errors='coerce')
fact = fact.dropna(subset=['Timestamp']).reset_index(drop=True)


rf = pd.read_csv('data/raw/rf_signal_data/rf_signal_data.csv')
rf['Timestamp'] = pd.to_datetime(rf['Timestamp'], format='mixed', dayfirst=False, errors='coerce')
rf['Interference Type'] = rf['Interference Type'].fillna('None')


interference_dist = rf['Interference Type'].value_counts(normalize=True)
bandwidth_mean = rf['Bandwidth'].mean()
bandwidth_std = rf['Bandwidth'].std()

print(f"Interference type distribution in RF data:\n{interference_dist}")


np.random.seed(42)
n = len(fact)

# RSRP proxy = Signal Strength (dBm) — already correct unit/scale
fact['rsrp_proxy'] = fact['Signal Strength (dBm)']

# SINR proxy = SNR — already correct concept
fact['sinr_proxy'] = fact['SNR']

# PRB utilization proxy = simulate congestion using call duration +
# a bandwidth-driven load factor sampled from RF dataset distribution
simulated_bandwidth = np.random.normal(bandwidth_mean, bandwidth_std, n)
simulated_bandwidth = np.clip(simulated_bandwidth, 50000, 20000000)
fact['prb_utilization'] = np.clip(
    (fact['Call Duration (s)'] / fact['Call Duration (s)'].max()) * 70
    + (1 - simulated_bandwidth / simulated_bandwidth.max()) * 30,
    0, 100
)

# Handover rate = how often a user's calls switch towers, per user
fact_sorted = fact.sort_values(['User ID', 'Timestamp']).copy()
fact_sorted['prev_tower'] = fact_sorted.groupby('User ID')['Tower ID'].shift(1)
fact_sorted['is_handover'] = (fact_sorted['Tower ID'] != fact_sorted['prev_tower']).astype(int)
handover_rate_by_user = fact_sorted.groupby('User ID')['is_handover'].mean()
fact['handover_rate'] = fact['User ID'].map(handover_rate_by_user).fillna(0)

# Assign interference type sampled from real RF dataset distribution
fact['interference_type'] = np.random.choice(
    interference_dist.index, size=n, p=interference_dist.values
)


def assign_degradation_cause(row):
    
    if row['Distance to Tower (km)'] > 6 and row['rsrp_proxy'] < -100:
        return 'distance'
    
    if row['interference_type'] != 'None' and row['sinr_proxy'] < 15:
        return 'interference'
  
    if row['prb_utilization'] > 60 and row['Call Duration (s)'] > 800:
        return 'congestion'
    
    if row['Attenuation'] > 12 and row['Distance to Tower (km)'] < 4:
        return 'hardware_fault'
    
    return 'congestion' if row['prb_utilization'] > 50 else 'distance'

fact['degradation_cause'] = fact.apply(assign_degradation_cause, axis=1)

print(f"\nDegradation cause distribution:")
print(fact['degradation_cause'].value_counts())


fact['tower_id'] = np.random.choice(towers['tower_id'], size=n)
towers_with_op = towers.merge(dim_operator, on=['operator', 'circle'], how='left')
tower_to_operator = towers_with_op.set_index('tower_id')['operator_id']
fact['operator_id'] = fact['tower_id'].map(tower_to_operator)


fact['date'] = fact['Timestamp'].dt.date
dim_date = pd.DataFrame({'full_date': fact['date'].unique()})
dim_date['date_id'] = dim_date.index + 1
dim_date['hour'] = 12  
dim_date['day_of_week'] = pd.to_datetime(dim_date['full_date']).dt.day_name()
dim_date.to_csv('data/processed/dim_date.csv', index=False)
fact = fact.merge(dim_date, left_on='date', right_on='full_date', how='left')


dim_environment = fact[['Environment']].drop_duplicates().reset_index(drop=True)
dim_environment['env_id'] = dim_environment.index + 1
dim_environment = dim_environment.rename(columns={'Environment': 'environment_type'})
dim_environment.to_csv('data/processed/dim_environment.csv', index=False)
fact = fact.merge(dim_environment, left_on='Environment', right_on='environment_type', how='left')


fact_final = fact[[
    'tower_id', 'operator_id', 'date_id', 'env_id',
    'rsrp_proxy', 'sinr_proxy', 'prb_utilization', 'handover_rate',
    'Distance to Tower (km)', 'Call Duration (s)', 'Attenuation',
    'interference_type', 'degradation_cause', 'Call Type', 'Incoming/Outgoing'
]].rename(columns={
    'Distance to Tower (km)': 'distance_to_tower',
    'Call Duration (s)': 'call_duration',
    'Attenuation': 'attenuation',
    'Call Type': 'call_type',
    'Incoming/Outgoing': 'call_direction'
})
fact_final['reading_id'] = fact_final.index + 1

fact_final.to_csv('data/processed/fact_rf_readings.csv', index=False)
print(f"\nfact_rf_readings shape: {fact_final.shape}")
print(fact_final.head(3))

print("\n All processed files saved to data/processed/")