import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 0. Create output folder
output_dir = "xgboost_output_CNratio_weather_mgmt"
os.makedirs(output_dir, exist_ok=True)

# 1. Load Data
df = pd.read_csv("/Users/utsabghimire/Downloads/SCINet/Updated_rye_datbase_format_all_data/Full_Data_With_Biomass_Mean_SD.csv")

# 2. Define Features
agronomic_cols = [
    "N_rate_fall.kg_ha", "N_rate_spring.kg_ha",
    "legume_preceding", "planting_method",
    "days_between_planting_and_sampling"
]
cat_features = ["legume_preceding", "planting_method"]
weather_cols = [col for col in df.columns if any(x in col for x in ["tmax", "tmin", "csrad", "crain", "gdd"])]
selected_features = [c for c in agronomic_cols + weather_cols if c in df.columns]

# 3. Filter and Prepare
df = df[df["shoot_CN_ratio"].notna()]
X = df[selected_features].copy()
y = df["shoot_CN_ratio"]

# Encode categorical variables
# One-hot encode categorical features

X[cat_features] = X[cat_features].fillna("missing").astype(str)
X = pd.get_dummies(X, columns=cat_features)



# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train Model
model = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=600,
    learning_rate=0.08,
    max_depth=6,
    verbosity=1
)
model.fit(X_train, y_train)

# 6. Predictions and Evaluation
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
percent_rmse = (rmse / y_test.mean()) * 100
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"%RMSE: {percent_rmse:.2f}%")
print(f"R²: {r2:.3f}")

# 7. Save predictions and actual vs. predicted plot
results_df = pd.DataFrame({"actual": y_test, "predicted": y_pred})
results_df.to_csv(f"{output_dir}/xgboost_CN_predictions.csv", index=False)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual C:N Ratio")
plt.ylabel("Predicted C:N Ratio")
plt.title("XGBoost - Actual vs Predicted C:N Ratio")
plt.tight_layout()
plt.savefig(f"{output_dir}/actual_vs_predicted_CN.png", dpi=300)
plt.close()

# 8. Feature Importance Plot
# ──────────────────────────────────────────
print("\n Extracting feature importances...")

booster = model.get_booster()
score_dict = booster.get_score(importance_type="gain")
fi_df = pd.DataFrame({
    "Feature Id": list(score_dict.keys()),
    "Importances": list(score_dict.values())
}).sort_values(by="Importances", ascending=False)

fi_df.head(30).to_csv(f"{output_dir}/feature_importance_CN_top30.csv", index=False)

plt.figure(figsize=(10, 8))
plt.barh(fi_df["Feature Id"][:30][::-1], fi_df["Importances"][:30][::-1])
plt.xlabel("Importance")
plt.title("Top 30 Feature Importances (C:N, XGBoost)")
plt.tight_layout()
plt.savefig(f"{output_dir}/feature_importance_CN_top30.png", dpi=300)
plt.close()

# 9. SHAP Summary Plot
# ──────────────────────────────────────────
print("\n Calculating SHAP values...")
explainer = shap.Explainer(model)
shap_values = explainer(X_train)

plt.figure()
shap.summary_plot(shap_values, X_train, show=False, max_display=30)
plt.tight_layout()
plt.savefig(f"{output_dir}/shap_summary_CN_top30.png", dpi=300)
plt.close()

print("\n✅ C:N Prediction Complete!")
