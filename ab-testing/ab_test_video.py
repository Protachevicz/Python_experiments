import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
from scipy.stats import norm

# Parameters for the A/B test
np.random.seed(42)
n_time_steps = 1000  # Number of hours
n_users_per_hour = 500  # Users per hour per group

# Function to simulate and generate frames
def generate_frames():
    # Initialize directories
    frames_dir = "frames"
    os.makedirs(frames_dir, exist_ok=True)

    # Parameters for two cases
    cases = [(0.15, 0.15), (0.152, 0.15)]
    data = []

    # Run simulations for both cases
    for true_rate_A, true_rate_B in cases:
        conversion_rates_A = []
        conversion_rates_B = []
        p_values = []

        cumulative_conversions_A = 0
        cumulative_conversions_B = 0
        cumulative_users_A = 0
        cumulative_users_B = 0

        for hour in range(1, n_time_steps + 1):
            # Simulate conversions for group A
            users_A = np.random.binomial(1, true_rate_A, n_users_per_hour)
            cumulative_conversions_A += np.sum(users_A)
            cumulative_users_A += n_users_per_hour

            # Simulate conversions for group B
            users_B = np.random.binomial(1, true_rate_B, n_users_per_hour)
            cumulative_conversions_B += np.sum(users_B)
            cumulative_users_B += n_users_per_hour

            # Calculate conversion rates
            current_conversion_rate_A = cumulative_conversions_A / cumulative_users_A
            current_conversion_rate_B = cumulative_conversions_B / cumulative_users_B

            conversion_rates_A.append(current_conversion_rate_A)
            conversion_rates_B.append(current_conversion_rate_B)

            # Calculate cumulative p-value using a z-test
            p_pooled = (cumulative_conversions_A + cumulative_conversions_B) / (cumulative_users_A + cumulative_users_B)
            se = np.sqrt(p_pooled * (1 - p_pooled) * (1 / cumulative_users_A + 1 / cumulative_users_B))
            z = (current_conversion_rate_A - current_conversion_rate_B) / se
            p_value = 2 * (1 - norm.cdf(abs(z)))  # Two-tailed p-value
            p_values.append((p_value, current_conversion_rate_A > current_conversion_rate_B))

        data.append((conversion_rates_A, conversion_rates_B, p_values))

    # Fixed limits for all plots
    conversion_rate_ylim = [0.14, 0.18]
    p_value_ylim = [0, 1]

    # Generate side-by-side plots
    for hour in range(1, n_time_steps + 1):
        fig, axs = plt.subplots(2, 2, figsize=(18, 12))

        # Left column: Case 1 (equal rates)
        axs[0, 0].plot(range(1, hour + 1), data[0][0][:hour], label="Group A", color="blue")
        axs[0, 0].plot(range(1, hour + 1), data[0][1][:hour], label="Group B", color="red")
        axs[0, 0].axhline(y=0.15, color="gray", linestyle="--", label="True Rate")
        axs[0, 0].set_ylim(conversion_rate_ylim)  # Fixed y-limits
        axs[0, 0].set_title(f"Equal Conversion Rates", fontsize=27)
        axs[0, 0].set_xlabel("Hours", fontsize=27)
        axs[0, 0].set_ylabel("Conversion Rate", fontsize=27)
        axs[0, 0].tick_params(axis='both', labelsize=22)
        axs[0, 0].legend(fontsize=22)
        axs[0, 0].grid()

        # P-value line in green and instant value in bold, larger text
        p_values_case1 = [p for p, is_A_winning in data[0][2][:hour]]
        axs[1, 0].plot(range(1, hour + 1), p_values_case1, color="green")
        axs[1, 0].text(hour, p_values_case1[-1] + 0.05, f"{p_values_case1[-1]:.4f}", color="green", fontsize=22, fontweight="bold", ha='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='green'))
        axs[1, 0].set_ylim(p_value_ylim)  # Fixed y-limits
        axs[1, 0].axhline(y=0.05, color="red", linestyle="--", label="Significance Threshold")
        axs[1, 0].set_title("Equal Rates - P-Value", fontsize=27)
        axs[1, 0].set_xlabel("Hours", fontsize=27)
        axs[1, 0].set_ylabel("P-Value", fontsize=27)
        axs[1, 0].tick_params(axis='both', labelsize=22)
        axs[1, 0].legend(fontsize=22)
        axs[1, 0].grid()

        # Right column: Case 2 (different rates)
        axs[0, 1].plot(range(1, hour + 1), data[1][0][:hour], label="Group A", color="blue")
        axs[0, 1].plot(range(1, hour + 1), data[1][1][:hour], label="Group B", color="red")
        axs[0, 1].axhline(y=0.15, color="gray", linestyle="--", label="True Rate B")
        axs[0, 1].axhline(y=0.152, color="gray", linestyle="--", label="True Rate A")
        axs[0, 1].set_ylim(conversion_rate_ylim)  # Fixed y-limits
        axs[0, 1].set_title(f"Different Conversion Rates", fontsize=27)
        axs[0, 1].set_xlabel("Hours", fontsize=27)
        axs[0, 1].set_ylabel("Conversion Rate", fontsize=27)
        axs[0, 1].tick_params(axis='both', labelsize=22)
        axs[0, 1].legend(fontsize=22)
        axs[0, 1].grid()

        # P-value line in green and instant value in bold, larger text
        p_values_case2 = [p for p, is_A_winning in data[1][2][:hour]]
        axs[1, 1].plot(range(1, hour + 1), p_values_case2, color="green")
        axs[1, 1].text(hour, p_values_case2[-1] + 0.05, f"{p_values_case2[-1]:.4f}", color="green", fontsize=22, fontweight="bold", ha='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='green'))
        axs[1, 1].set_ylim(p_value_ylim)  # Fixed y-limits
        axs[1, 1].axhline(y=0.05, color="red", linestyle="--", label="Significance Threshold")
        axs[1, 1].set_title("Different Rates - P-Value", fontsize=27)
        axs[1, 1].set_xlabel("Hours", fontsize=27)
        axs[1, 1].set_ylabel("P-Value", fontsize=27)
        axs[1, 1].tick_params(axis='both', labelsize=22)
        axs[1, 1].legend(fontsize=22)
        axs[1, 1].grid()

        # Add hour counter
        fig.suptitle(f"Hour {hour}", fontsize=30, fontweight="bold")

        # Save frame
        frame_path = os.path.join(frames_dir, f"frame_{hour:03d}.png")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(frame_path)
        plt.close()

    return frames_dir

# Generate frames
frames_dir = generate_frames()

# Create the video using FFmpeg
video_path = "ab_test_side_by_side.mp4"
result = subprocess.run([
    "ffmpeg", "-y", "-framerate", "10", "-i", f"{frames_dir}/frame_%03d.png",
    "-c:v", "libx264", "-pix_fmt", "yuv420p", video_path
], capture_output=True, text=True)

if result.returncode != 0:
    print("FFmpeg failed with the following error:")
    print(result.stderr)
else:
    print(f"Video saved as {video_path}")

# Clean up temporary frames
for file in os.listdir(frames_dir):
    os.remove(os.path.join(frames_dir, file))
os.rmdir(frames_dir)

