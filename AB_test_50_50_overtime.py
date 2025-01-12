import numpy as np
import matplotlib.pyplot as plt

# Parameters for the simulation
n_iterations = 1000  # Number of iterations (time steps)
true_prob_A = 0.5    # True preference for A (50%)
true_prob_B = 0.5    # True preference for B (50%)
sample_size_per_step = 50  # Number of samples per step

# Lists to store cumulative results
cumulative_A = []
cumulative_B = []
cumulative_preference_A = []
cumulative_preference_B = []

# Simulate A/B test over time
total_A = 0
total_B = 0
for i in range(1, n_iterations + 1):
    # Generate random results for A and B
    results_A = np.random.binomial(1, true_prob_A, sample_size_per_step)
    results_B = np.random.binomial(1, true_prob_B, sample_size_per_step)
    
    # Update cumulative counts
    total_A += results_A.sum()
    total_B += results_B.sum()
    
    # Store cumulative preferences
    cumulative_A.append(total_A)
    cumulative_B.append(total_B)
    cumulative_preference_A.append(total_A / (total_A + total_B))
    cumulative_preference_B.append(total_B / (total_A + total_B))

# Plot results with increased font size
plt.figure(figsize=(12, 6))
plt.plot(range(1, n_iterations + 1), cumulative_preference_A, label='Preference for A', alpha=0.8)
plt.plot(range(1, n_iterations + 1), cumulative_preference_B, label='Preference for B', alpha=0.8)
plt.axhline(y=true_prob_A, color='gray', linestyle='--', label='True Preference (50%)')
plt.title('A/B Test Over Time: Fluctuations and Convergence', fontsize=18)
plt.xlabel('Iterations (Time)', fontsize=16)
plt.ylabel('Cumulative Preference', fontsize=16)
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.show()
