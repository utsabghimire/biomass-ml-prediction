import pandas as pd
import os

# ──────────────────────────────────
# 1. Load the dataset
# ──────────────────────────────────
data_path = "/Users/utsabghimire/Downloads/SCINet/Updated_rye_datbase_format_all_data/Full_Data_With_Biomass_Mean_SD.csv"  # Change if needed
df = pd.read_csv(data_path)

# ──────────────────────────────────
# 2. Define growth stages and week ranges
# ──────────────────────────────────
growth_stage_definitions = {
    "Seedling": list(range(1, 5)),          # Week 1–4
    "Tillering": list(range(5, 13)),          # Week 5–12
    "Jointing": list(range(13, 16)),           # Week 13–15
    "Booting": list(range(16, 20)),           # Week 16–19
    "Maturity": list(range(20, 34)),          # Week 20–33
}

# ──────────────────────────────────
# 3. Aggregate weather data by stage
# ──────────────────────────────────
stage_agg_df = pd.DataFrame(index=df.index)

for stage, weeks in growth_stage_definitions.items():
    tmin_cols = [f"week_{w}_tmin" for w in weeks if f"week_{w}_tmin" in df.columns]
    tmax_cols = [f"week_{w}_tmax" for w in weeks if f"week_{w}_tmax" in df.columns]
    csrad_cols = [f"week_{w}_csrad" for w in weeks if f"week_{w}_csrad" in df.columns]
    crain_cols = [f"week_{w}_crain" for w in weeks if f"week_{w}_crain" in df.columns]
    
    stage_agg_df[f"{stage}_tmin_avg"] = df[tmin_cols].mean(axis=1)
    stage_agg_df[f"{stage}_tmax_avg"] = df[tmax_cols].mean(axis=1)
    stage_agg_df[f"{stage}_csrad_sum"] = df[csrad_cols].sum(axis=1)
    stage_agg_df[f"{stage}_crain_sum"] = df[crain_cols].sum(axis=1)

# Add mean_biomass column to the aggregated DataFrame
if "mean_biomass" in df.columns:
    stage_agg_df["mean_biomass"] = df["mean_biomass"]

# ──────────────────────────────────
# 4. Save output
# ──────────────────────────────────
output_path = "aggregated_weather_by_stage.csv"
stage_agg_df.to_csv(output_path, index=False)
print(f"\n✅ Aggregated data saved to: {output_path}")
