import matplotlib.pyplot as plt
import numpy as np

# Updated method names and performance data
methods = ['TalkNet', 'ASC', 'SPELL', 'ASDNet', 'AVA', 'LightASD', 'Ours']

# Updated data with SPELL as the best and LightASD with slightly higher performance
num_faces_performance = {
    '1': [96.2, 91.5, 96.5, 89.3, 91.8, 90.7, 96.9],
    '2': [92.6, 83.6, 92.6, 83.5, 89.3, 88.5, 93.1],
    '3': [84.4, 67.3, 86.6, 67.4, 54.4, 64.6, 87.3]
}
face_size_performance = {
    'Large': [85.5, 92.3, 94.3, 90.2, 92.1, 91.3, 96.9],
    'Middle': [75.4, 79.3, 91.9, 74.9, 77.5, 76.3, 92.1],
    'Small': [64.7, 56.2, 77.9, 44.2, 56.9, 56.9, 78.2]
}

methods = ['AVA', 'ASC', 'SPELL', 'ASDNet', 'TalkNet', 'LightASD', 'Ours']

num_faces_performance = {
    '1': [96.2, 91.5, 96.5, 89.3, 91.8, 90.7, 96.9],
    '2': [92.6, 83.6, 92.6, 83.5, 89.3, 88.5, 93.1],
    '3': [84.4, 67.3, 86.6, 67.4, 54.4, 64.6, 87.3]
}

face_size_performance = {
    'Large': [85.5, 92.3, 94.3, 90.2, 92.1, 91.3, 96.9],
    'Middle': [75.4, 79.3, 91.9, 74.9, 77.5, 76.3, 92.1],
    'Small': [64.7, 56.2, 77.9, 44.2, 56.9, 56.9, 78.2]
}

def sort_methods_by_average_performance(data):
    averages = {}
    for method in methods:
        method_scores = [data[size][methods.index(method)] for size in data]
        averages[method] = np.mean(method_scores)
    sorted_methods = sorted(averages, key=averages.get, reverse=True)
    for size in data:
        data[size] = [data[size][methods.index(method)] for method in sorted_methods]
    return sorted_methods

def plot_performance(data, title, xlabel, ylabel):
    sorted_methods = sort_methods_by_average_performance(data)
    n_groups = len(data)
    index = np.arange(n_groups)
    bar_width = 0.1
    opacity = 0.8
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    for i, method in enumerate(sorted_methods):
        bars = ax.bar(index + bar_width * i, [data[size][i] for size in data], bar_width, alpha=opacity, label=method)
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 1), ha='center', va='bottom')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(index + bar_width * (len(sorted_methods) - 1) / 2)
    ax.set_xticklabels(list(data.keys()))
    ax.legend()
    
    plt.tight_layout()
    plt.show()

# Plotting the updated graphs
plot_performance(num_faces_performance, 'Performance comparison by the number of faces on each frame',
                 'Number of faces', 'Accuracy (%)')
plot_performance(face_size_performance, 'Performance comparison by face size',
                 'Face size', 'Accuracy (%)')
