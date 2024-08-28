import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import seaborn as sns

# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Construct the full path to the merged_output.json file
json_file_path = os.path.join(script_dir, 'merged_output.json')

# Load the merged JSON data from the file
with open(json_file_path, 'r') as file:
    json_data = json.load(file)

# Extract parameters and target values
params = []
targets = []
param_names = None

for entry in json_data:
    if param_names is None:
        param_names = list(entry['params'].keys())
    params.append(list(entry['params'].values()))
    targets.append(entry['target'])

# Convert to numpy arrays
X = np.array(params)
y = np.array(targets)

# Apply PCA to reduce the dimensions to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Identify the indices of targets that are 0
zero_target_indices = np.where(y == 0)[0]

# Identify the indices of the top 20 non-zero targets
non_zero_indices = np.where(y != 0)[0]
top_20_indices = np.argsort(y[non_zero_indices])[-20:]
top_20_indices = non_zero_indices[top_20_indices]
best_index = top_20_indices[-1]

# Create a KDE plot
plt.figure(figsize=(10, 7))
sns.kdeplot(x=X_pca[:, 0], y=X_pca[:, 1], fill=True, cmap='viridis')

# Plot all points with smaller dots
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=10, edgecolor='k', label='All Data Points')

# Highlight the top 20 non-zero targets with bigger dots
plt.scatter(X_pca[top_20_indices, 0], X_pca[top_20_indices, 1], c='red', s=100, edgecolor='k', label='Top 20 mAPs')

# Mark the best non-zero target with a cross
plt.scatter(X_pca[best_index, 0], X_pca[best_index, 1], c='blue', s=200, edgecolor='k', marker='x', linewidths=3, label='Best mAP')

# Plot the zero targets with white dots
plt.scatter(X_pca[zero_target_indices, 0], X_pca[zero_target_indices, 1], c='white', s=50, edgecolor='k', label='Target = 0')

# Add legend to the plot
plt.legend(loc='upper right')

plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('PCA of Bayesian Search Parameters (KDE with Top and Zero mAPs)')
plt.show()


for i, param_name in enumerate(param_names):
    plt.figure(figsize=(8, 6))

    # Filter to get targets >= 93
    high_target_indices = np.where(y >= 93)[0]
    X_high = X[high_target_indices, i].reshape(-1, 1)
    y_high = y[high_target_indices]

    # Plot targets >= 93
    plt.scatter(X_high, y_high, c='blue', edgecolor='k', s=5, label='mAP >= 93')

    # Fit a quadratic regression model
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X_high)
    model = LinearRegression()
    model.fit(X_poly, y_high)
    y_pred = model.predict(X_poly)

    # Plot the quadratic line of best fit
    sort_indices = np.argsort(X_high[:, 0])
    plt.plot(X_high[sort_indices], y_pred[sort_indices], color='black', linewidth=2, label='Quadratic Best Fit')

    # Bin the attribute data to identify ranges with lots of zero targets
    bins = np.linspace(X[:, i].min(), X[:, i].max(), 20)
    zero_hist, _ = np.histogram(X[zero_target_indices, i], bins=bins)

    # Increase threshold for many zeros and plot an indicator at y=93
    high_zero_count = zero_hist > 10  # Increased threshold for many zeros
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    plt.scatter(bin_centers[high_zero_count], np.full_like(bin_centers[high_zero_count], 93), c='red', s=50, marker='|', label='Many Zero mAPs')

    # Highlight the top 20 targets in a different color and smaller size
    top_20_indices = np.argsort(y)[-20:]  # Get indices of the top 20 targets
    X_top20 = X[top_20_indices, i].reshape(-1, 1)
    y_top20 = y[top_20_indices]
    plt.scatter(X_top20, y_top20, c='green', edgecolor='k', s=20, label='Top 20 mAP')  # Smaller size

    # Add a vertical line at the x of the best target
    best_target_index = np.argmax(y)  # Index of the best (maximum) target
    best_x = X[best_target_index, i]

    # Determine the number of significant figures needed
    if best_x < 10:
        best_x_str = f'{best_x:.3g}'  # Use 3 significant figures for small values
    else:
        best_x_str = f'{best_x:.2f}'  # Use 2 decimal places for larger values

    plt.axvline(x=best_x, color='green', linewidth=1, linestyle='--')

    # Add green text for the value of the top target on the x-axis
    plt.text(best_x, plt.ylim()[0], best_x_str, color='green', fontsize=10, ha='center', va='bottom')

    plt.xlabel(param_name)
    plt.ylabel('Final mAP')
    plt.title(f'Scatter Plot of {param_name} vs. mAP (Showing mAP >= 93)')
    plt.grid(True)

    # Adjust legend position to be inside the plot on the middle right with a transparent box
    plt.legend(loc='center right', bbox_to_anchor=(1, 0.5), fancybox=True, framealpha=0.5, edgecolor='k')

    plt.show()
