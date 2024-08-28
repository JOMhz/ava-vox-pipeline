import matplotlib.pyplot as plt
import numpy as np

# Data for the plot
models = ['AVA', 'GraVi-T', 'ASDNet', 'ASC', 'TalkNet', 'LightASD', 'VoxID']
num_faces = ['1', '2', '3']
face_number_data = np.array([
    [87.9, 96.8, 93.3, 95.4, 95.7, 96.2, 96.8],
    [71.6, 93.8, 85.8, 89.6, 92.4, 92.6, 94.1],
    [54.4, 85.6, 68.2, 80.3, 83.7, 84.4, 87.2]
])

face_sizes = ['Large', 'Middle', 'Small']
face_size_data = np.array([
    [86.4, 96.9, 93.0, 95.3, 96.3, 96.5, 96.9],
    [68.3, 92.5, 79.4, 85.9, 89.8, 91.2, 93.4],
    [44.9, 79.2, 55.2, 63.7, 74.3, 77.5, 79.7]
])

bar_width = 0.13
index = np.arange(len(num_faces))

# Colors for each model
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#17becf']


# Calculate the average rank for each model across the number of faces
average_rank = np.mean(face_number_data, axis=0)
sorted_indices = np.argsort(-average_rank)  # Sorting in descending order

# Sort the data and models based on the average rank
sorted_data = face_number_data[:, sorted_indices]
sorted_models = [models[i] for i in sorted_indices]
sorted_colors = [colors[i] for i in sorted_indices]

# Plot the sorted bar chart
fig, ax = plt.subplots(figsize=(10, 6))

for i in range(sorted_data.shape[1]):
    ax.bar(index + i * bar_width, sorted_data[:, i], bar_width, label=sorted_models[i], color=sorted_colors[i])

for i in range(sorted_data.shape[1]):
    for j in range(sorted_data.shape[0]):
        ax.text(index[j] + i * bar_width, sorted_data[j, i] + 1, f'{sorted_data[j, i]:.1f}', ha='center', va='bottom', fontsize=10)

# Customize the plot
ax.set_xlabel('Number of faces', fontsize=12)
ax.set_ylabel('mAP(%)', fontsize=12)
ax.set_title('mAP(%) for Different Models and Number of Faces (Sorted by Average Rank)', fontsize=14)
ax.set_xticks(index + bar_width * (sorted_data.shape[1] / 2 - 0.5))
ax.set_xticklabels(num_faces)
ax.set_ylim(0, 110)
ax.legend(title='Models', fontsize=10, title_fontsize=12, loc='upper right')

plt.show()

# Calculate the average rank for each model across the face sizes
average_rank_new = np.mean(face_size_data, axis=0)
sorted_indices_new = np.argsort(-average_rank_new)  # Sorting in descending order

# Sort the data and models based on the average rank
sorted_data_new = face_size_data[:, sorted_indices_new]
sorted_models_new = [models[i] for i in sorted_indices_new]
sorted_colors_new = [colors[i] for i in sorted_indices_new]

# Plot the sorted bar chart
fig, ax = plt.subplots(figsize=(10, 6))

for i in range(sorted_data_new.shape[1]):
    ax.bar(index + i * bar_width, sorted_data_new[:, i], bar_width, label=sorted_models_new[i], color=sorted_colors_new[i])

for i in range(sorted_data_new.shape[1]):
    for j in range(sorted_data_new.shape[0]):
        ax.text(index[j] + i * bar_width, sorted_data_new[j, i] + 1, f'{sorted_data_new[j, i]:.1f}', ha='center', va='bottom', fontsize=10)

# Customize the plot
ax.set_xlabel('Face size', fontsize=12)
ax.set_ylabel('mAP(%)', fontsize=12)
ax.set_title('mAP(%) for Different Models and Face Sizes (Sorted by Average Rank)', fontsize=14)
ax.set_xticks(index + bar_width * (sorted_data_new.shape[1] / 2 - 0.5))
ax.set_xticklabels(face_sizes)
ax.set_ylim(0, 110)
ax.legend(title='Models', fontsize=10, title_fontsize=12, loc='upper right')

plt.show()