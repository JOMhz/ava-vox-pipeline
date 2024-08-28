import pickle
import numpy as np
from collections import defaultdict
import os
import sys
import pandas as pd
import time

# ensemble_val_res_path = 'exps/ensemble/val_res.csv'
# ensemble_train_res_path = 'exps/ensemble/train_res.csv'
# ensemble_pd = pd.read_csv(ensemble_train_res_path, nrows=5)

# ensemble_pd : video_id  frame_timestamp  entity_box_x1  entity_box_y1  entity_box_x2  entity_box_y2             label                entity_id     score
# There is over 1 million rows. I want to covert data frame into dictionaries by video id so one dictionary per video_id
# The keys of each dictionary is the time stamps that match the video_id
# Each timestamp key will contain a list of dictionaries
# Each dictionary within that list will contain 'person_box': 'entity_box_x1,entity_box_y1,entity_box_x2,entity_box_y2', 'person_id': 'entity_id', 'feat': []
# feat will contain an np.array with each score_x not including the column 'score


# Assuming the CSV file has been read as follows:


def convert_timestamp(timestamp):
    if timestamp.is_integer():
        return int(timestamp)
    else:
        return timestamp

def group_to_dict(group, video_id, total_groups, ensemble_pd):
    result = defaultdict(list)
    start_time = time.time()
    
    # Get a list of columns to keep in 'feat', excluding 'score'
    feat_cols = [col for col in ensemble_pd.columns if 'score' in col and col != 'score']
    
    for index, row in group.iterrows():
        timestamp = convert_timestamp(row['frame_timestamp'])
        timestamp_str = str(timestamp)
        person_box = ','.join([
            str(row['entity_box_x1']),
            str(row['entity_box_y1']),
            str(row['entity_box_x2']),
            str(row['entity_box_y2'])
        ])
        features = np.array(row[feat_cols].values, dtype=np.float32)
        entry = {
            'person_box': person_box,
            'person_id': row['entity_id'],
            'feat': features
        }
        result[timestamp_str].append(entry)
    return result


for file in ['val', 'train']:
    ensemble_res_path = 'exps/ensemble/' + file + '_res.csv'
    ensemble_pd = pd.read_csv(ensemble_res_path)
    # Directory to save pickle files
    save_directory = "AVADataPath/LSTM/" + file
    os.makedirs(save_directory, exist_ok=True)
    # Group DataFrame by 'video_id' and apply the function to each group
    start_time = time.time()
    total_groups = len(ensemble_pd.groupby('video_id'))
    processed_groups = 0
    for idx, (video_id, group) in enumerate(ensemble_pd.groupby('video_id'), 1):
        video_dict = group_to_dict(group, video_id, total_groups, ensemble_pd)
        if video_dict is not None:
            save_path = os.path.join(save_directory, f"{video_id}.pkl")
            with open(save_path, "wb") as f:
                pickle.dump(video_dict, f)
            processed_groups += 1
            elapsed_time = time.time() - start_time
            remaining_time = (total_groups - processed_groups) * (elapsed_time / processed_groups)
            print(f"\rProcessed {processed_groups}/{total_groups} groups. Elapsed time: {elapsed_time:.2f} seconds. Estimated time remaining: {remaining_time:.2f} seconds.", end='')

    end_time = time.time()
    processing_time = end_time - start_time
    print(f"\nAll pickle files saved successfully. Processing time: {processing_time:.2f} seconds.")