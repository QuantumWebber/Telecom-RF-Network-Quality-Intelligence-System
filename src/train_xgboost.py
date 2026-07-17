
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib


df = pd.read_csv('data/processed/fact_rf_readings.csv')
print(f"Loaded {len(df)} rows")
print(f"\nClass distribution:\n{df['degradation_cause'].value_counts()}")


feature_cols = [
    'rsrp_proxy', 'sinr_proxy', 'prb_utilization', 'handover_rate',
    'distance_to_tower', 'call_duration', 'attenuation'
]
X = df[feature_cols].copy()

# Encode target
le = LabelEncoder()
y = le.fit_transform(df['degradation_cause'])
print(f"\nClasses: {list(le.classes_)}")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")


print("\nApplying SMOTE to balance training classes...")
smote = SMOTE(random_state=42, k_neighbors=min(3, X_train.shape[0]-1))
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"After SMOTE - train size: {len(X_train_res)}")
print(pd.Series(y_train_res).value_counts())


print("\nTraining XGBoost...")
model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=len(le.classes_),
    max_depth=5,
    learning_rate=0.1,
    n_estimators=200,
    eval_metric='mlogloss',
    random_state=42
)
model.fit(X_train_res, y_train_res)


y_pred = model.predict(X_test)
print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))


importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print("\nFEATURE IMPORTANCE")
print(importance)


joblib.dump(model, 'data/processed/xgb_degradation_model.pkl')
joblib.dump(le, 'data/processed/label_encoder.pkl')
print("\nModel saved to data/processed/xgb_degradation_model.pkl")