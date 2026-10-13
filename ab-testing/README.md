# A/B Testing Simulations

Three experiments that simulate, analyze, and visualize A/B tests, exploring how statistical significance behaves under different scenarios: equal vs. different conversion rates, varying user segments, and random noise over time.

## Projects in this folder

### 1. `ab_test_video.py`
Simulates two parallel A/B tests over 1000 "hours" of traffic (500 users/hour per group): one where both variants have the **same true conversion rate** (0.15 vs 0.15) and one where they **genuinely differ** (0.152 vs 0.15). For each hour it computes the cumulative conversion rate per group and a two-tailed z-test p-value, then renders a frame with four synchronized plots (conversion rates and p-value evolution for both scenarios). The frames are stitched into an MP4 with FFmpeg.

**Purpose:** demonstrates visually why peeking at an A/B test too early is dangerous, the p-value fluctuates wildly before stabilizing, and can cross the significance threshold (0.05) by chance even when there is no real effect.

**Requirements:** `numpy`, `matplotlib`, `scipy`, and `ffmpeg` installed and available on PATH.

**Run:** `python ab_test_video.py` → produces `ab_test_side_by_side.mp4`.

### 2. `ab_test_by_condition.py`
Simulates an A/B test segmented into five audience conditions, from "Strong A" (clear preference for variant A) to "Strong B" (clear preference for variant B), via "Neutral" in the middle. For each condition it simulates binomial conversions for both groups, runs a Welch's t-test, and plots conversion rates by condition and group.

**Purpose:** illustrates how the same A/B test can have a different (or even reversed) winner depending on the audience segment, a core idea behind segmented/stratified experimentation and Simpson's-paradox-style pitfalls in naive pooled analysis.

**Requirements:** `numpy`, `pandas`, `matplotlib`, `scipy`. Note: uses `ace_tools` for the results tables, which is only available in certain notebook environments; replace with `print(results)` / `print(t_test_results)` to run standalone.

### 3. `ab_test_50_50_convergence.py`
A minimal simulation where both variants have an identical true conversion probability (50/50). Over 1000 iterations it tracks the cumulative preference for each variant and plots how the observed split converges toward the true 50/50 split as more samples accumulate.

**Purpose:** a clean, teachable example of the Law of Large Numbers and why small-sample A/B results are noisy even with no real difference between variants.

**Requirements:** `numpy`, `matplotlib`.
