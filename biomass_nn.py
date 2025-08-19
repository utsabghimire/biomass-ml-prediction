import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
import os

# --------------------------------------------------
# 1. LOAD & FILTER DATA
# --------------------------------------------------
df = pd.read_csv("/Users/utsabghimire/Downloads/SCINet/Updated_rye_datbase_format_all_data/July26_Omit_Yes_667_Rows_with_Biomass_and_CN_Ratio_Averaged.csv")

input_features = [
    "growing_days", "N_rate_fall.kg_ha", "N_rate_spring.kg_ha",
    "GS0_20avgTavg", "GS0_20avgSrad", "GS0_20cRain", "GS0_20cGDD",
    "GS20_30avgTavg", "GS20_30avgSrad", "GS20_30cRain", "GS20_30cGDD", "zone",
    "FallcumGDD", "SpringcumGDD", "OM (%/100)", "Sand", "Silt", "Clay",
    "legume_preceding", "planting_method"
]
biomass_col = "biomass_mean"
df = df[input_features + [biomass_col]].dropna()
df = df[df[biomass_col] >= 100]

X = df[input_features]
y = df[biomass_col]

# --------------------------------------------------
# 2. PREPROCESS INPUTS (NO NORMALIZATION)
# --------------------------------------------------
cat_cols = ["zone", "legume_preceding", "planting_method"]
num_cols = [col for col in input_features if col not in cat_cols]

preprocessor = ColumnTransformer([
    ("num", "passthrough", num_cols),
    ("cat", OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

# --------------------------------------------------
# 3. BUILD NEURAL NETWORK (NO DROPOUT/REGULARIZATION)
# --------------------------------------------------
model = models.Sequential([
    layers.Input(shape=(X_train_proc.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mae',
    metrics=['mae']
)

# --------------------------------------------------
# 4. TRAINING
# --------------------------------------------------
early_stop = callbacks.EarlyStopping(patience=10, restore_best_weights=True)

history = model.fit(
    X_train_proc, y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)

# --------------------------------------------------
# 5. EVALUATION
# --------------------------------------------------
y_pred = model.predict(X_test_proc).flatten()

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n✅ Final Neural Net:")
print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.3f}")

# --------------------------------------------------
# 6. SAVE PREDICTIONS
# --------------------------------------------------
results_df = pd.DataFrame({
    "actual_biomass": y_test.values,
    "predicted_biomass": y_pred
})
output_dir = "NN_Biomass_Results"
os.makedirs(output_dir, exist_ok=True)
results_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

# --------------------------------------------------
# 7. PREDICTED VS ACTUAL PLOT
# --------------------------------------------------
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolor='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Biomass (kg/ha)")
plt.ylabel("Predicted Biomass (kg/ha)")
plt.title("Neural Network: Predicted vs Actual Biomass")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "pred_vs_actual.png"))
plt.show()
