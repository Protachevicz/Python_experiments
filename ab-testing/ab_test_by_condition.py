import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Set a random seed for reproducibility
np.random.seed(42)

# Define the conditions and their respective preferences
conditions = ["Strong A", "Weak A", "Neutral", "Weak B", "Strong B"]
sample_sizes = [500, 400, 300, 400, 500]  # Number of participants per condition
conversion_rates_A = [0.8, 0.6, 0.5, 0.4, 0.2]  # Strong A prefers A
conversion_rates_B = [0.2, 0.4, 0.5, 0.6, 0.8]  # Strong B prefers B

# Simulate A/B test data
data = []
for condition, size, rate_A, rate_B in zip(conditions, sample_sizes, conversion_rates_A, conversion_rates_B):
    group_A = np.random.binomial(1, rate_A, size)  # Simulated conversions for Group A
    group_B = np.random.binomial(1, rate_B, size)  # Simulated conversions for Group B
    data.append(pd.DataFrame({
        "Condition": condition,
        "Group": "A",
        "Conversion": group_A
    }))
    data.append(pd.DataFrame({
        "Condition": condition,
        "Group": "B",
        "Conversion": group_B
    }))

# Combine all data into a single DataFrame
df = pd.concat(data)

# Analyze results: calculate conversion rates and differences
results = df.groupby(["Condition", "Group"])["Conversion"].mean().unstack()
results["Difference"] = results["B"] - results["A"]
results["Winner"] = np.where(results["Difference"] > 0, "B", "A")

# Perform t-tests for each condition
t_tests = []
for condition in conditions:
    group_A = df[(df["Condition"] == condition) & (df["Group"] == "A")]["Conversion"]
    group_B = df[(df["Condition"] == condition) & (df["Group"] == "B")]["Conversion"]
    t_stat, p_value = ttest_ind(group_A, group_B, equal_var=False)  # Welch's t-test
    t_tests.append({"Condition": condition, "T-Statistic": t_stat, "P-Value": p_value})

t_test_results = pd.DataFrame(t_tests)

# Visualization: Bar plot of conversion rates
results[["A", "B"]].plot(kind="bar", figsize=(10, 6))
plt.title("Conversion Rates by Condition and Group")
plt.ylabel("Conversion Rate")
plt.xticks(rotation=0)
plt.legend(title="Group")
plt.tight_layout()
plt.show()

# Display results in tables
import ace_tools as tools; tools.display_dataframe_to_user(name="A/B Test Results by Condition", dataframe=results)
tools.display_dataframe_to_user(name="T-Test Results by Condition", dataframe=t_test_results)
