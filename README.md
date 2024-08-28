# ava-vox-pipeline
The repo contains the official code and models for "Audio-Visual Active Speaker Detection: A Hybrid Approach Using Machine Learning"
![voxID](https://github.com/user-attachments/assets/936b6385-cdc8-4118-8cdb-16b358972bf1)
## Notes Before Starting

To avoid issues with GPU drivers you should do all work within a docker container https://hub.docker.com/r/nvidia/cuda

## Dependencies

Start by building the environment

```
conda create -n VoxID python=3.10.12
conda activate VoxID
pip install -r requirement.txt

## VoxID in AVA-Activespeaker dataset

#### Data preparation

The following script can be used to download and prepare the AVA dataset for training.

The file size of the videos before pre-processing is 96.6 GB
You will need 110GB free before you run this function.


```
python setup.py
```

#### Feature Extraction

I have use clip_length=11 as suggested by https://github.com/fuankarion/active-speakers-context and https://github.com/SRA2/SPELL
Make sure the same clip length is used for both.
```
python Encoder/STE_train.py clip_length cuda_device_number
python Encoder/STE_forward.py clip_length cuda_device_number
```

#### Stage-2 - Intermediate Decision
```
python LSTMs/train.py --dataPathAVA AVADataPath
python LSTMs/train.py --dataPathAVA AVADataPath --evaluation
python LSTMs/ensemble_pkl.py
```
exps/exps1/score.txt: output score file, exps/exp1/model/model_00xx.model: trained model, exps/exps1/val_res.csv: prediction for val set.

#### Building Spatial-Temporal Graph
python generate_graph.py --feature resnet18-tsm-aug


#### Training GNN
python train_val.py --feature resnet18-tsm-aug
