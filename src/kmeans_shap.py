
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


df = pd.read_csv('data/processed/fact_rf_readings.csv')


cluster_features = ['rsrp_proxy', 'sinr_proxy', 'prb_utilization', 'handover_rate']
X_cluster = df[cluster_features].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

 
print("Finding optimal number of clusters...")
best_k, best_score = 2, -1
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f"  k={k}: silhouette score = {score:.3f}")
    if score > best_score:
        best_k, best_score = k, score

print(f"\nBest k = {best_k} (silhouette = {best_score:.3f})")

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['tower_cluster'] = kmeans.fit_predict(X_scaled)

print("\nCluster profiles (mean values):")
cluster_profile = df.groupby('tower_cluster')[cluster_features].mean()
print(cluster_profile)

print("\nCluster sizes:")
print(df['tower_cluster'].value_counts().sort_index())


def label_cluster(row):
    if row['rsrp_proxy'] < -100 and row['sinr_proxy'] < 15:
        return 'Poor coverage zone'
    elif row['prb_utilization'] > 60:
        return 'Congested tower'
    elif row['handover_rate'] > 0.6:
        return 'High mobility zone'
    else:
        return 'Healthy tower'

cluster_profile['label'] = cluster_profile.apply(label_cluster, axis=1)
print("\nCluster labels:")
print(cluster_profile[['label']])

df.to_csv('data/processed/fact_rf_readings_clustered.csv', index=False)
joblib.dump(kmeans, 'data/processed/kmeans_model.pkl')
joblib.dump(scaler, 'data/processed/kmeans_scaler.pkl')


print("\n" + "="*60)
print("SHAP EXPLAINABILITY")
print("="*60)

model = joblib.load('data/processed/xgb_degradation_model.pkl')
le = joblib.load('data/processed/label_encoder.pkl')

feature_cols = [
    'rsrp_proxy', 'sinr_proxy', 'prb_utilization', 'handover_rate',
    'distance_to_tower', 'call_duration', 'attenuation'
]
X = df[feature_cols].copy()

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)


plt.figure()
if isinstance(shap_values, list):
    
    shap_avg = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    shap.summary_plot(shap_avg, X, feature_names=feature_cols, show=False)
else:
    shap.summary_plot(shap_values, X, feature_names=feature_cols, show=False)
plt.tight_layout()
plt.savefig('data/processed/shap_summary.png', dpi=120)
print("SHAP summary plot saved to data/processed/shap_summary.png")

print("\nDone! K-Means + SHAP complete.")