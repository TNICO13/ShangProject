import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm


def forward_stepwise_anova_formula(data, response, predictors, significance_level=0.05):
    """
    Perform forward stepwise regression using nested-model ANOVA tests with a formula interface.

    At each iteration, the current model's R² is printed.

    Parameters:
        data (pd.DataFrame): DataFrame containing the data.
        response (str): The response variable name.
        predictors (list of str): List of candidate predictor strings to be used in the formula.
                                  Categorical variables should already be wrapped as C(var).
        significance_level (float): p-value threshold for adding a predictor.

    Returns:
        final_model: The final fitted statsmodels OLS model.
        selected: List of predictors included in the final model.
    """
    # Start with an intercept-only model.
    selected = []
    remaining = predictors.copy()
    current_formula = f"{response} ~ 1"
    current_model = smf.ols(current_formula, data=data).fit()

    iteration = 1
    improved = True

    # Print initial R²
    print(f"Iteration {iteration} (Intercept only): R² = {current_model.rsquared:.4f}")

    while improved and remaining:
        print(f"\nIteration {iteration + 1}:")
        print("Currently selected predictors:", selected)

        candidate_results = []

        # Evaluate adding each remaining predictor.
        for predictor in remaining:
            formula_candidate = f"{response} ~ " + " + ".join(selected + [predictor])
            try:
                candidate_model = smf.ols(formula_candidate, data=data).fit()
                # Compare candidate model with the current model using ANOVA.
                anova_results = anova_lm(current_model, candidate_model)
                # The second row corresponds to the additional predictor; extract its p-value.
                p_value = anova_results["Pr(>F)"][1]
                candidate_results.append((p_value, predictor, candidate_model))
            except Exception as e:
                print(f"Error testing addition of {predictor}: {e}")
                continue

        if not candidate_results:
            print("No candidate predictors were successfully evaluated. Terminating forward selection.")
            break

        # Sort candidate additions by p-value in ascending order (lowest p-value first).
        candidate_results.sort(key=lambda x: x[0])
        best_p, best_predictor, best_model = candidate_results[0]

        # If the best candidate has a p-value below the threshold, add it.
        if best_p < significance_level:
            selected.append(best_predictor)
            remaining.remove(best_predictor)
            current_model = best_model
            print(f"Added predictor: {best_predictor} (addition p-value = {best_p:.4f})")
            print(f"Updated R² = {current_model.rsquared:.4f}")
        else:
            print("No candidate predictor meets the significance threshold. Stopping forward selection.")
            improved = False

        iteration += 1

    return current_model, selected


# -----------------------------
# Load the imputed, z-score standardized DataFrame.
# -----------------------------
df = pd.read_pickle("static_dataframe_imputed_zscore.pkl")

# -----------------------------
# Define the target variable and predictors.
# -----------------------------
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
    "pl_bmassprov"
    # If you have additional categorical predictors, wrap them as needed.
]
wrapped_categorical = [f"C({var})" for var in categorical_predictors]

# Combined predictor list.
predictor_list = numeric_predictors + wrapped_categorical

# -----------------------------
# Drop rows with missing values for all model variables.
# -----------------------------
model_vars = [target] + numeric_predictors + categorical_predictors
df_model = df.dropna(subset=model_vars)
print(f"\nNumber of rows used for modeling: {len(df_model)}")

# -----------------------------
# Perform forward stepwise selection using nested-model ANOVA.
# -----------------------------
print("\nStarting Forward Stepwise ANOVA Model Selection:")
final_model, selected_predictors = forward_stepwise_anova_formula(
    data=df_model,
    response=target,
    predictors=predictor_list,
    significance_level=0.05
)

print("\nFinal Selected Predictors:")
print(selected_predictors)
print("\nFinal Model Summary:")
print(final_model.summary())
