import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm


def backward_stepwise_anova_formula(data, response, predictors, significance_level=0.05):
    """
    Perform backward stepwise regression using nested-model ANOVA tests with a formula interface.

    At each iteration, this function prints the current model's R².

    Parameters:
        data (pd.DataFrame): DataFrame containing the data.
        response (str): The response variable name.
        predictors (list of str): List of predictor strings to be used in the formula.
                                  Categorical variables should already be wrapped as C(var).
        significance_level (float): p-value threshold for removal.

    Returns:
        final_model: The final fitted statsmodels OLS model.
        selected: List of remaining predictors in the final model.
    """
    # Start with the full model including all candidate predictors.
    selected = predictors.copy()
    current_formula = f"{response} ~ " + " + ".join(selected)
    current_model = smf.ols(current_formula, data=data).fit()

    print(f"Iteration 1: Initial model R² = {current_model.rsquared:.4f}")

    iteration = 2  # Start next iteration numbering from 2
    improved = True
    while improved and len(selected) > 0:
        print(f"\nIteration {iteration}: Current predictors:")
        print(selected)
        candidate_results = []

        # Evaluate removal of each predictor individually.
        for predictor in selected:
            candidate_predictors = [p for p in selected if p != predictor]
            if candidate_predictors:
                formula_candidate = f"{response} ~ " + " + ".join(candidate_predictors)
            else:
                formula_candidate = f"{response} ~ 1"
            try:
                candidate_model = smf.ols(formula_candidate, data=data).fit()
                # Compare candidate (reduced) model with the current (full) model using ANOVA.
                anova_results = anova_lm(candidate_model, current_model)
                # The second row corresponds to the comparison – extract its p-value.
                p_value = anova_results["Pr(>F)"][1]
                candidate_results.append((p_value, predictor, candidate_model))
            except Exception as e:
                print(f"Error testing removal of {predictor}: {e}")
                continue

        if not candidate_results:
            print("No candidates were successfully evaluated. Terminating stepwise elimination.")
            break

        # Sort the candidate removals by p-value (largest first).
        candidate_results.sort(key=lambda x: x[0], reverse=True)
        best_p, best_predictor, best_model = candidate_results[0]

        if best_p > significance_level:
            print(f"Removing predictor: {best_predictor} (removal p-value = {best_p:.4f})")
            selected.remove(best_predictor)
            current_model = best_model
            print(f"Updated R² = {current_model.rsquared:.4f}")
        else:
            print("No candidate removal has a p-value above the threshold. Stopping elimination.")
            improved = False

        iteration += 1

    return current_model, selected


# -----------------------------
# Load the imputed, z-score standardized DataFrame.
# -----------------------------
df = pd.read_pickle("static_dataframe_imputed_zscore.pkl")

# -----------------------------
# Define the response variable and predictors.
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

# Categorical predictors (wrapped for dummy coding):
categorical_predictors = [
    "pl_bmassprov",
    "st_spectype",
    "st_metratio"
]
wrapped_categorical = [f"C({var})" for var in categorical_predictors]

# Combined predictor list.
predictor_list = numeric_predictors + wrapped_categorical

# -----------------------------
# Drop rows with missing values for all model variables.
# -----------------------------
model_vars = [target] + numeric_predictors + categorical_predictors
df_model = df.dropna(subset=model_vars)
print(f"Number of rows used for modeling: {len(df_model)}")

# -----------------------------
# Fit the full model and perform backward stepwise ANOVA.
# -----------------------------
print("\nStarting Backward Stepwise ANOVA Model Selection:")
final_model, final_predictors = backward_stepwise_anova_formula(
    data=df_model,
    response=target,
    predictors=predictor_list,
    significance_level=0.05
)

print("\nFinal Selected Predictors:")
print(final_predictors)
print("\nFinal Model Summary:")
print(final_model.summary())
