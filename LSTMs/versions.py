import sys
import pkg_resources

# List of packages to check
packages = ['torch', 'torchvision', 'numpy', 'python_speech_features']

# Print Python version
print(f"Python version: {sys.version}")

# Print each package version
for package in packages:
    version = pkg_resources.get_distribution(package).version
    print(f"{package} version: {version}")

# CUDA version (if applicable)
try:
    import torch
    print(f"CUDA version: {torch.version.cuda}")
except AttributeError:
    print("CUDA is not installed or torch is not installed correctly.")
