import pandas as pd
import statsmodels.formula.api as smf

# -----------------------------
# Load the imputed, z-score standardized DataFrame.
# -----------------------------
df = pd.read_pickle("static_dataframe_imputed_zscore.pkl")

# -----------------------------
# Define variables to be used in the model.
# -----------------------------
# We use all variables except the planet name (which is textual) as candidate predictors.
# From your df.dtypes output the following are available:
#   - Numerical predictors (int64 or float64): sy_snum, pl_controv_flag, pl_orbper, pl_orbsmax,
#       pl_rade, pl_radj, pl_bmasse, pl_bmassj, pl_orbeccen, pl_eqt, ttv_flag, st_teff, st_rad,
#       st_mass, st_met, st_logg, ra, dec, sy_dist, sy_vmag, sy_kmag, sy_gaiamag.
#   - Categorical predictors (object): pl_bmassprov, st_spectype, st_metratio.
#
# We set "sy_pnum" as the target.
target = "st_mass"

# Numeric predictors:
numeric_predictors = [
    "sy_pnum",
    "sy_snum",
    "pl_controv_flag",
    "pl_orbper",
    "pl_orbsmax",
    "pl_rade",
    "pl_radj",
    "pl_bmasse",
    "pl_bmassj",
    "pl_orbeccen",
    "pl_eqt",
    "ttv_flag",
    "st_teff",
    "st_rad",
    "st_met",
    "st_logg",
    "ra",
    "dec",
    "sy_dist",
    "sy_vmag",
    "sy_kmag",
    "sy_gaiamag"
]

# Categorical predictors:
categorical_predictors = [
    "pl_bmassprov",
    "st_spectype",
    "st_metratio"
]

# -----------------------------
# Create a combined list of predictors.
# -----------------------------
# For numeric predictors, include them directly;
# for categorical predictors, wrap them in C().
all_predictors = numeric_predictors + [f"C({var})" for var in categorical_predictors]

# Build the regression formula.
formula = target + " ~ " + " + ".join(all_predictors)
print("Regression Formula:")
print(formula)

# -----------------------------
# Drop rows with missing values in any variable used in the model.
# -----------------------------
# We'll drop any row that has missing data among the target or the predictors.
model_vars = [target] + numeric_predictors + categorical_predictors
df_model = df.dropna(subset=model_vars)
print(f"\nNumber of rows used for modeling: {len(df_model)}")

# -----------------------------
# Fit the linear regression model using statsmodels OLS.
# -----------------------------
model = smf.ols(formula, data=df_model).fit()

# -----------------------------
# Print the detailed regression summary.
# -----------------------------
print("\nLinear Regression Model Summary:")
print(model.summary())
