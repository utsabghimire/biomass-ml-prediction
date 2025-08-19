import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os

# --------------------------------------------------
# 1. Load Data
# --------------------------------------------------
df = pd.read_csv("/Users/utsabghimire/Downloads/SCINet/Updated_rye_datbase_format_all_data/July24_727_rows.csv")

input_features = [
    "growing_days", "N_rate_fall.kg_ha", "N_rate_spring.kg_ha",
    "GS0_20avgTavg", "GS0_20avgSrad", "GS0_20cRain", "GS0_20cGDD",
    "GS20_30avgTavg", "GS20_30avgSrad", "GS20_30cRain", "GS20_30cGDD", "zone",
    "FallcumGDD", "SpringcumGDD", "OM (%/100)", "Sand", "Silt", "Clay",
    "legume_preceding", "planting_method"
]
target = "biomass_mean"

df = df[input_features + [target]].dropna()
df = df[df[target] >= 100]

X = df[input_features]
y = df[target]

# --------------------------------------------------
# 2. Preprocess
# --------------------------------------------------
cat_cols = ["zone", "legume_preceding", "planting_method"]
num_cols = [col for col in input_features if col not in cat_cols]

preprocessor = ColumnTransformer([
    ("num", "passthrough", num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_proc = preprocessor.fit_transform(X_train).astype(np.float32)
X_test_proc = preprocessor.transform(X_test).astype(np.float32)
y_train = y_train.values.astype(np.float32)
y_test = y_test.values.astype(np.float32)

# --------------------------------------------------
# 3. PyTorch Dataloaders
# --------------------------------------------------
train_dataset = TensorDataset(torch.from_numpy(X_train_proc), torch.from_numpy(y_train))
test_dataset = TensorDataset(torch.from_numpy(X_test_proc), torch.from_numpy(y_test))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# --------------------------------------------------
# 4. Define Model
# --------------------------------------------------
class BiomassNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze()

model = BiomassNN(X_train_proc.shape[1])
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# --------------------------------------------------
# 5. Training Loop
# --------------------------------------------------
loss_fn = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
best_loss = float('inf')
patience, wait = 10, 0

for epoch in range(1, 201):
    model.train()
    train_losses = []
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    avg_train_loss = np.mean(train_losses)
    print(f"Epoch {epoch:03d}: MAE Loss = {avg_train_loss:.2f}")
    
    if avg_train_loss < best_loss:
        best_loss = avg_train_loss
        wait = 0
        torch.save(model.state_dict(), "best_model.pt")
    else:
        wait += 1
        if wait >= patience:
            print("⏹️ Early stopping")
            break

# --------------------------------------------------
# 6. Evaluate
# --------------------------------------------------
model.load_state_dict(torch.load("best_model.pt"))
model.eval()
with torch.no_grad():
    X_test_tensor = torch.from_numpy(X_test_proc).to(device)
    y_pred = model(X_test_tensor).cpu().numpy()

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n✅ PyTorch NN Performance:")
print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.3f}")

# --------------------------------------------------
# 7. Save Predictions & Plot
# --------------------------------------------------
os.makedirs("PyTorch_NN_Results", exist_ok=True)
pd.DataFrame({
    "actual_biomass": y_test,
    "predicted_biomass": y_pred
}).to_csv("PyTorch_NN_Results/predictions.csv", index=False)

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel("Actual Biomass")
plt.ylabel("Predicted Biomass")
plt.title("PyTorch NN: Predicted vs Actual Biomass")
plt.tight_layout()
plt.savefig("PyTorch_NN_Results/pred_vs_actual.png")
plt.show()
