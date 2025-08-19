"""
CatBoost Regression for Cover Crop Biomass Prediction
----------------------------------------------------

This script trains a CatBoost regression model to predict cover crop biomass
using agronomic, soil, and weather features. CatBoost natively handles
categorical variables and provides convenient utilities for model training
and evaluation. The code closely follows the feature definitions used in
the neural network scripts (`biomass_nn.py` and `biomass_ptorch.py`).

Update the `data_path` variable to point to your local CSV file. The
dataset should include the features listed in `input_features` and a
`biomass_mean` column representing the target variable.

Dependencies:
  - pandas
  - numpy
  - scikit-learn
  - catboost
  - matplotlib

Run this script as a standalone module:

```
python biomass_catboost.py
```
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor, Pool
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Load Data
# --------------------------------------------------
# TODO: Set this path to your CSV file with biomass measurements.
data_path = "path/to/your/cover_crop_dataset.csv"
df = pd.read_csv(data_path)

# Specify input features used for biomass prediction. These mirror the
# features used in the neural network scripts. Feel free to adjust or
# augment this list based on your data availability and domain knowledge.
input_features = [
    "growing_days", "N_rate_fall.kg_ha", "N_rate_spring.kg_ha",
    "GS0_20avgTavg", "GS0_20avgSrad", "GS0_20cRain", "GS0_20cGDD",
    "GS20_30avgTavg", "GS20_30avgSrad", "GS20_30cRain", "GS20_30cGDD",
    "zone", "FallcumGDD", "SpringcumGDD", "OM (%/100)",
    "Sand", "Silt", "Clay", "legume_preceding", "planting_method"
]

# Target column for biomass (kg/ha)
target = "biomass_mean"

# Drop rows with missing values in the selected columns and remove
# extremely low biomass samples (optional).
df = df[input_features + [target]].dropna()
df = df[df[target] >= 100]

X = df[input_features]
y = df[target]

# Identify categorical feature names (CatBoost handles these directly)
cat_cols = ["zone", "legume_preceding", "planting_method"]
cat_indices = [X.columns.get_loc(col) for col in cat_cols]

# --------------------------------------------------
# 2. Train/Test Split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create CatBoost Pool objects (optional but recommended)
train_pool = Pool(data=X_train, label=y_train, cat_features=cat_indices)
test_pool = Pool(data=X_test, label=y_test, cat_features=cat_indices)

# --------------------------------------------------
# 3. Train CatBoost Model
# --------------------------------------------------
# Hyperparameters can be tuned using CatBoost's built-in tools or
# external libraries such as Optuna. The parameters below offer a
# reasonable starting point.
model = CatBoostRegressor(
    iterations=600,
    learning_rate=0.05,
    depth=6,
    loss_function="MAE",
    eval_metric="RMSE",
    verbose=50,
    random_seed=42
)

model.fit(train_pool, eval_set=test_pool)

# --------------------------------------------------
# 4. Evaluate Performance
# --------------------------------------------------
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n✅ CatBoost Model Performance:")
print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.3f}")

# --------------------------------------------------
# 5. Save Results and Feature Importance
# --------------------------------------------------
output_dir = "CatBoost_Biomass_Results"
os.makedirs(output_dir, exist_ok=True)

# Save predictions
pd.DataFrame({
    "actual_biomass": y_test,
    "predicted_biomass": y_pred
}).to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

# Plot Actual vs. Predicted
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolor='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Biomass (kg/ha)")
plt.ylabel("Predicted Biomass (kg/ha)")
plt.title("CatBoost: Predicted vs Actual Biomass")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "pred_vs_actual.png"), dpi=300)
plt.close()

# Feature importance using CatBoost built-in method
feature_importances = model.get_feature_importance()
fi_df = pd.DataFrame({
    "feature": X.columns,
    "importance": feature_importances
}).sort_values(by="importance", ascending=False)
fi_df.to_csv(os.path.join(output_dir, "feature_importances.csv"), index=False)

print("\nScript complete. Results saved to:", output_dir)
