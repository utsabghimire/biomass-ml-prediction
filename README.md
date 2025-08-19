# Biomass & C:N Ratio Prediction for Cover Crops

This repository contains a collection of machine learning scripts for predicting
cover crop biomass and shoot C:N ratio using agronomic, soil and weather
features.  It provides ready-to-run models based on tree ensembles and deep
neural networks, along with data processing utilities for aggregating
time-series weather data into biologically meaningful growth stages.

## ✨ Project Highlights

* **Multiple modeling approaches:** leverage gradient boosted trees (XGBoost
  & CatBoost) and fully connected neural networks (TensorFlow/Keras and
  PyTorch) to compare performance and interpretability.
* **Rich feature set:** integrate agronomic inputs (N rates, planting
  intervals), soil composition, cumulative growing degree days and weekly
  weather summaries such as temperature, solar radiation and rainfall.
* **Interpretability:** generate feature importances and SHAP summary plots
  for the tree models to understand key drivers of biomass and C:N ratio.
* **Modular scripts:** each model lives in its own script and can be run
  independently.  Outputs (predictions, plots and feature rankings) are
  saved in organized folders.

## 📦 Repository Structure

```
.
├── CN_Ratio_XGBoost.py        # XGBoost model for shoot C:N ratio prediction
├── biomass_nn.py              # Keras neural network for biomass prediction
├── biomass_ptorch.py          # PyTorch neural network for biomass prediction
├── biomass_catboost.py        # CatBoost regression for biomass prediction
├── extract_stage_weather.py   # Aggregate weekly weather into growth stages
├── requirements.txt           # Python dependencies for all scripts
├── .gitignore                 # Ignore intermediate files/outputs
└── README.md                  # Project description and usage (this file)
```

## 🧮 Features

Each script consumes a CSV file containing experimental observations from
cover crop trials.  The following feature categories are used:

* **Agronomic:** days between planting and sampling, fall and spring N rates
  (`N_rate_fall.kg_ha`, `N_rate_spring.kg_ha`), presence of a legume
  preceding the cover crop, planting method (drill vs broadcast), number of
  days the crop was growing, etc.
* **Soil:** organic matter (`OM (%/100)`), sand, silt and clay
  percentages.
* **Weather:** weekly or stage-based summaries of minimum and maximum
  temperatures, cumulative solar radiation and rainfall (`tmin`, `tmax`,
  `csrad`, `crain`), as well as cumulative growing degree days (GDD).
* **Cumulative metrics:** total GDD in fall and spring (`FallcumGDD`,
  `SpringcumGDD`), and counts of growing days (`growing_days`).
* **Categorical:** geographic zone, whether a legume preceded the cover crop,
  and planting method.  These are automatically handled via one-hot
  encoding (for neural networks & XGBoost) or passed as categorical indices
  for CatBoost.

## 🛠️ Scripts & Models

### `CN_Ratio_XGBoost.py`

Predicts shoot C:N ratio using an XGBoost regression model.  The script:

1. Reads a CSV file (update the path at the top of the script).
2. Selects agronomic and weather features and one-hot encodes categorical
   variables.
3. Splits the data into train/test sets (80/20).
4. Trains an `xgboost.XGBRegressor` with chosen hyperparameters.
5. Evaluates the model on the test set (RMSE, MAE, percent RMSE and R²).
6. Saves a CSV of predictions, an actual vs. predicted scatter plot and a
   bar chart of the top 30 feature importances.  A SHAP summary plot is
   also produced for more detailed interpretability.

### `biomass_nn.py`

Trains a fully connected neural network using TensorFlow/Keras to predict
biomass.  Key aspects:

* Uses the same feature set defined above, with categorical variables
  one-hot encoded via `ColumnTransformer`.
* Builds a simple feedforward architecture (128→64→32 hidden units) with
  ReLU activations.
* Employs early stopping (patience of 10 epochs) to avoid overfitting.
* Saves predictions and generates a scatter plot of actual vs. predicted
  biomass.

### `biomass_ptorch.py`

Provides an alternative implementation of the neural network using PyTorch.
It mirrors the Keras model architecture and training loop, including early
stopping based on validation loss.  Predictions, evaluation metrics and
plots are saved in a `PyTorch_NN_Results` folder.

### `biomass_catboost.py`

Introduces a CatBoost regression model, which natively handles categorical
features without requiring one-hot encoding.  This script uses the same
feature definitions as the neural network models and prints RMSE, MAE and
R², while exporting predictions and feature importances.

### `extract_stage_weather.py`

Aggregates weekly weather variables into developmental stages (seedling,
tillering, jointing, booting and maturity).  For each stage it computes
average minimum and maximum temperatures and sums of solar radiation and
rainfall.  The resulting stage-level features can be merged back into your
main dataset for model training.

## 🚀 Getting Started

1. **Clone the repo** (or download the source files).
2. **Install dependencies:**

   ```bash
   python -m pip install -r requirements.txt
   ```

3. **Prepare your data:** ensure your CSV file includes the columns listed
   under *Features*.  Modify the `data_path` variables at the top of each
   script to point to your dataset.  The provided paths are examples.
4. **Run the desired model:** for example,

   ```bash
   python CN_Ratio_XGBoost.py      # predict shoot C:N ratio
   python biomass_nn.py            # Keras neural network for biomass
   python biomass_ptorch.py        # PyTorch neural network for biomass
   python biomass_catboost.py      # CatBoost regression for biomass
   python extract_stage_weather.py # build stage-based weather features
   ```

Outputs (predictions, plots, feature importance tables) will be saved in
 the respective directories created by each script (e.g. `xgboost_output_CNratio_weather_mgmt/`,
 `NN_Biomass_Results/`, etc.).

## 📝 Requirements

Dependencies are listed in `requirements.txt`.  At a minimum, you'll need:

- Python ≥ 3.8
- pandas, numpy, scikit-learn
- xgboost, catboost, shap (for interpretability)
- tensorflow (for Keras), torch (for PyTorch)
- matplotlib

Consider creating a virtual environment before installing packages to avoid
conflicts with your system Python:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
python -m pip install -r requirements.txt
```

## 📚 References & Notes

The data used in these scripts originate from rye cover crop experiments
conducted by USDA-ARS and collaborators.  Weather data were aggregated
using the `extract_stage_weather.py` utility.  Feel free to adapt these
scripts to your own datasets and research questions.

Feature engineering ideas, hyperparameters and modeling approaches were
inspired by Utsab’s work on predicting biomass and C:N ratio using
machine learning.  Pull requests are welcome—please open an issue if
you’d like to contribute or report a problem.

## 📜 License

This project is licensed under the MIT License—see the `LICENSE` file for
 details.
