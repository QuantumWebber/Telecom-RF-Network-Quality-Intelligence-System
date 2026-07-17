
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)


df = pd.read_csv('data/processed/fact_rf_readings_clustered.csv')

print("Building hourly aggregated time-series for LSTM...")
n_hours = 24 * 30 
hours = pd.date_range('2023-01-01', periods=n_hours, freq='h')


base_prb = df['prb_utilization'].mean()
prb_std = df['prb_utilization'].std()
base_sinr = df['sinr_proxy'].mean()

hour_of_day = hours.hour
daily_pattern = 15 * np.sin((hour_of_day - 9) * np.pi / 12) ** 2  # peak load 9am-9pm

prb_series = np.clip(
    base_prb + daily_pattern + np.random.normal(0, prb_std * 0.3, n_hours),
    0, 100
)
sinr_series = base_sinr - (prb_series / 100) * 8 + np.random.normal(0, 2, n_hours)


call_drop_prob = np.clip(
    (prb_series / 100) * 0.6 + (1 - (sinr_series - sinr_series.min()) /
    (sinr_series.max() - sinr_series.min())) * 0.4 + np.random.normal(0, 0.03, n_hours),
    0, 1
)

ts = pd.DataFrame({
    'timestamp': hours,
    'prb_utilization': prb_series,
    'sinr_proxy': sinr_series,
    'call_drop_prob': call_drop_prob
})

print(f"Time series shape: {ts.shape}")
print(ts.head())



LOOKBACK = 24
HORIZON = 24

scaler = MinMaxScaler()
scaled = scaler.fit_transform(ts[['prb_utilization', 'sinr_proxy', 'call_drop_prob']])

X, y = [], []
for i in range(len(scaled) - LOOKBACK - HORIZON):
    X.append(scaled[i:i+LOOKBACK])
    y.append(scaled[i+LOOKBACK+HORIZON-1, 2])  

X, y = np.array(X), np.array(y)
print(f"\nSequences: X={X.shape}, y={y.shape}")

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


print("\nTraining LSTM...")
model = keras.Sequential([
    layers.Input(shape=(LOOKBACK, 3)),
    layers.LSTM(32, return_sequences=True),
    layers.Dropout(0.2),
    layers.LSTM(16),
    layers.Dropout(0.2),
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=16,
    verbose=1
)


test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest MAE (scaled 0-1): {test_mae:.4f}")

y_pred = model.predict(X_test).flatten()
print(f"\nSample predictions vs actual (call-drop probability):")
for i in range(5):
    print(f"  Predicted: {y_pred[i]:.3f}  |  Actual: {y_test[i]:.3f}")

# Plot training history
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title('Loss'); plt.legend()
plt.subplot(1, 2, 2)
plt.plot(y_test[:100], label='actual')
plt.plot(y_pred[:100], label='predicted')
plt.title('Call-drop probability: actual vs predicted (24h ahead)')
plt.legend()
plt.tight_layout()
plt.savefig('data/processed/lstm_results.png', dpi=120)
print("\nPlot saved to data/processed/lstm_results.png")

model.save('data/processed/lstm_calldrop_model.keras')
print("Model saved!")