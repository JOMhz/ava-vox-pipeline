## Getting Started (Action Localization)
### Annotations
Download the annotations of AVA-Actions from the official site:
```
DATA_DIR="data/annotations"

wget https://research.google.com/ava/download/ava_val_v2.2.csv.zip -P ${DATA_DIR}
wget https://research.google.com/ava/download/ava_action_list_v2.2_for_activitynet_2019.pbtxt -P ${DATA_DIR}

unzip ${DATA_DIR}/ava_val_v2.2.csv.zip -d ${DATA_DIR}
mv ${DATA_DIR}/research/action_recognition/ava/website/www/download/ava_val_v2.2.csv ${DATA_DIR}
```

### Features
Download `SLOWFAST-64x2-R101.zip` from the Google Drive link from [SPELL](https://github.com/SRA2/SPELL#code-usage) and unzip under `data/features`.
> We use the features from the thirdparty repositories. SLOWFAST-64x2-R101 is obtained by using the official code of [SlowFast](https://github.com/facebookresearch/SlowFast) with the pretrained checkpoint ([SLOWFAST_64x2_R101_50_50.pkl](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/ava/SLOWFAST_64x2_R101_50_50.pkl)) in [SlowFast Model Zoo](https://github.com/facebookresearch/SlowFast/blob/main/MODEL_ZOO.md).

### Directory Structure
The data directories should look as follows:
```
|-- data
    |-- annotations
        |-- ava_val_v2.2.csv
        |-- ava_action_list_v2.2_for_activitynet_2019.pbtxt
    |-- features
        |-- SLOWFAST-64x2-R101
            |-- train
            |-- val
```

### Experiments
We can perform the experiments on action localization with the default configuration by following the three steps below.

#### Step 1: Graph Generation
Run the following command to generate spatial-temporal graphs from the features:
```
python data/generate_spatial-temporal_graphs.py --features SLOWFAST-64x2-R101 --ec_mode cdi --time_span 90 --tau 3
```
The generated graphs will be saved under `data/graphs`. Each graph captures long temporal context information in a video, which spans about 90 seconds (specified by `--time_span`).

#### Step 2: Training
Next, run the training script by passing the default configuration file:
```
python tools/train_context_reasoning.py --cfg configs/action-localization/ava_actions/SPELL_default.yaml
```
The results and logs will be saved under `results`.

#### Step 3: Evaluation
Now, we can evaluate the trained model's performance:
```
python tools/evaluate.py --exp_name SPELL_AL_default --eval_type AVA_AL
```
This will print the evaluation score.
