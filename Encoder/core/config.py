import sys

import torch.nn as nn
import torch.optim as optim
import core.models as mdet

STE_inputs = {
    # input files
    'csv_train_full': 'training_data/ava_activespeaker_train_augmented.csv',
    'csv_val_full': 'training_data/ava_activespeaker_val_augmented.csv',

    # Data config
    'audio_dir': '/home/jacobmaschler/Documents/Light-ASD-main/AVADataPath/clips_audios/',
    'video_dir': '/home/jacobmaschler/Documents/Light-ASD-main/AVADataPath/clips_videos/',
    'models_out': 'results/',
    'threads': 4
}

ASC_inputs = {
    # input files
    'features_train_full': '.../train_forward/*.csv',
    'features_val_full': '.../val_forward/*.csv',

    # Data config
    'models_out': '...'
}

ASC_inputs_forward = {
    # input files
    'features_train_full': '...',
    'features_val_full': '.../val_forward'
}

#Optimization params
STE_optimization_params = {
    # Net Arch
    'backbone': mdet.resnet18_two_streams,

    # Optimization config
    'optimizer': optim.Adam,
    'criterion': nn.CrossEntropyLoss(),
    'learning_rate': 3e-4,
    'epochs': 100,
    'step_size': 40,
    'gamma': 0.1,

    # Batch Config
    'batch_size': 64,
    'threads': 4
}

STE_forward_params = {
    # Net Arch
    'backbone': mdet.resnet18_two_streams_forward,

    # Batch Config
    'batch_size': 1,
    'threads': 1
}

ASC_optimization_params = {
    # Optimizer config
    'optimizer': optim.Adam,
    'criterion': nn.CrossEntropyLoss(),
    'learning_rate': 3e-6,
    'epochs': 15,
    'step_size': 10,
    'gamma': 0.1,

    # Batch Config
    'batch_size': 64,
    'threads': 0
}

ASC_forward_params = {
    # Batch Config
    'batch_size': 1,
    'threads': 1
}
