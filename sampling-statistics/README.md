# Sampling Distributions & the Central Limit Theorem

### `clt_sampling_distribution.py`
A hands-on demonstration of the **Central Limit Theorem** using real-world EPA air quality data (`c4_epa_air_quality.csv`, AQI readings).

**What it does:**
- Computes the true population mean AQI
- Draws 10,000 random samples (with replacement, n=50) and computes the sample mean for each, building an empirical sampling distribution
- Computes the standard error of the sample mean
- Plots a histogram of the 10,000 sample means overlaid with the theoretical normal curve predicted by the CLT, marking the population mean, a single sample's mean, and the mean of all sample means

**Purpose:** a teaching example showing that even when the underlying data isn't normally distributed, the distribution of sample means approaches a normal distribution as sample size grows, the foundation behind confidence intervals and hypothesis testing used throughout the A/B testing experiments in this repository.

**Requirements:** `numpy`, `pandas`, `matplotlib`, `statsmodels`, `scipy`. Needs a local `c4_epa_air_quality.csv` file (EPA air quality dataset) in the same folder, add your own or point `pd.read_csv` to your dataset.

**Note:** this is adapted from a guided exercise; a couple of lines reference a `sampled_data` variable that should be defined via `epa_data.sample(n=50, random_state=...)` before running end-to-end.
