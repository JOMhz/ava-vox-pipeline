import argparse
import os
import subprocess
import pandas as pd
import cv2
import glob
from scipy.io import wavfile
from utils import init_args

# Define the directory where your CSV files are stored
csvs_folder_names = ['video_audio_csvs_val', 'video_audio_csvs_train']

# Define the column names since the CSV doesn't have headers
column_names = [
    'video_id', 'frame_timestamp', 'entity_box_x1', 'entity_box_y1',
    'entity_box_x2', 'entity_box_y2', 'label', 'entity_id'
]

def get_video_fps(video_id):
    video_extensions = ['.mkv', '.mp4', '.webm']
    base_path = 'ava_dataset/'  # Adjust this path to your videos directory
    
    for ext in video_extensions:
        video_path = base_path + video_id + ext
        cam = cv2.VideoCapture(video_path)
        if cam.isOpened():  # Successfully opened the video file
            fps = cam.get(cv2.CAP_PROP_FPS)
            cam.release()  # Don't forget to release the video capture object
            return fps
    
    # If no video file was successfully opened
    print(f"No video file found for video_id: {video_id} with tried extensions: {video_extensions}")
    return None


def process_all_csvs(input_folder, processed_output_file, combined_output_file, video_ids_output_file):
    final_data = []  # List to hold all processed data
    combined_data = []
    video_ids = []
    
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.csv'):
            file_path = os.path.join(input_folder, file_name)
            df = pd.read_csv(file_path, header=None, names=column_names)

            processed_data = {}  # Dictionary to store processed data for the current CSV

            for _, row in df.iterrows():
                entity_id = row['entity_id']
                video_id = row['video_id']
                frame_timestamp = row['frame_timestamp']
                label_id = 1 if row['label'] == 'SPEAKING_AUDIBLE' else 0
                entity_box_x1 = row['entity_box_x1']
                entity_box_y1 = row['entity_box_y1']
                entity_box_x2 = row['entity_box_x2']
                entity_box_y2 = row['entity_box_y2']
                label = row['label']

                if entity_id not in processed_data:
                    processed_data[entity_id] = {
                        'video_id': video_id,
                        'frame_timestamps': [],
                        'label_ids': [],
                    }

                processed_data[entity_id]['frame_timestamps'].append(frame_timestamp)
                processed_data[entity_id]['label_ids'].append(label_id)
                combined_data.append({
                    'video_id': video_id,
                    'frame_timestamp': frame_timestamp,
                    'entity_box_x1': entity_box_x1,
                    'entity_box_y1': entity_box_y1,
                    'entity_box_x2': entity_box_x2,
                    'entity_box_y2': entity_box_y2,
                    'label': label,
                    'entity_id': entity_id,
                    'label_id': label_id
                })
            
            video_fps = get_video_fps(video_id)
            video_ids.append(video_id)

            for entity_id, info in processed_data.items():
                final_data.append({
                    'entity_id': entity_id,
                    'number_of_frames': len(info['frame_timestamps']),
                    'average_frame_rate': video_fps,
                    'label_ids': info['label_ids'],
                })
    combined_df = pd.DataFrame(combined_data)
    combined_df.to_csv(combined_output_file, index=False)
    
    processed_df = pd.DataFrame(final_data)
    processed_df.to_csv(processed_output_file, index=False, header=False, sep='\t')

    video_ids_df = pd.DataFrame(video_ids, columns=['video_id'])
    video_ids_df.to_csv(video_ids_output_file, index=False, header=False)

def process_csv_folder_list(csvs_folder_names):
    # Process CSVs in each folder and save to a corresponding output file
    for folder_name in csvs_folder_names:
        input_folder = folder_name
        processed_output_file = f'ava_training_data/transformed_csvs/{folder_name.split("_")[-1]}.csv'
        combined_output_file = f'ava_training_data/transformed_csvs/combined_{folder_name.split("_")[-1]}.csv'
        video_ids_output_file = f'ava_training_data/transformed_csvs/video_ids_{folder_name.split("_")[-1]}.csv'
        print(f'Processing data from: {input_folder}')
        process_all_csvs(input_folder, processed_output_file, combined_output_file, video_ids_output_file)
        print(f'Processed data saved to: {processed_output_file}')
        print(f'Combined data saved to: {combined_output_file}')


def preprocess_AVA(args):
    """
    This pre-processing code is inspired by https://github.com/TaoRuijie/TalkNet-ASD/blob/main/utils/tools.py
    Which in turn was inspired by https://github.com/fuankarion/active-speakers-context/tree/master/data
    """
    if not args:
        args = argparse.Namespace()
        args = init_args()
    extract_audio(args) # Take 1 hour
    extract_audio_clips(args) # Takes 3 minutes
    extract_face_clips_from_videos(args) # Takes about 1 days


def extract_audio(args):
    # Extracts the audio from videos, processing takes approximately 1 hour
    for data_type in ['train', 'val']:
        input_folder = f'{args.visual_orig_path_AVA}/{data_type}'
        output_folder = f'{args.audio_orig_path_AVA}/{data_type}'
        os.makedirs(output_folder, exist_ok=True)
        videos = glob.glob(f"{input_folder}/*")
        
        for video_path in videos:
            audio_path = f'{output_folder}/{os.path.basename(video_path).split(".")[0]}.wav'
            cmd = f"ffmpeg -y -i {video_path} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads 8 {audio_path} -loglevel panic"
            subprocess.call(cmd, shell=True, stdout=None)

def extract_audio_clips(args):
    # Extracts audio clips from the audio, processing takes approximately 3 minutes
    data_type_map = {'train': 'train', 'val': 'val'}
    for data_type in ['train', 'val']:
        df = pd.read_csv(os.path.join(args.trial_path_AVA, f'combined_{data_type}.csv'), engine='python')
        df_neg = pd.concat([df[df['label_id'] == 0], df[df['label_id'] == 2]])
        df_pos = df[df['label_id'] == 1]
        df = pd.concat([df_pos, df_neg]).reset_index(drop=True)
        df = df.sort_values(['entity_id', 'frame_timestamp']).reset_index(drop=True)
        
        entity_list = df['entity_id'].unique()
        grouped_df = df.groupby('entity_id')
        audio_features = {}

        output_dir = os.path.join(args.audio_path_AVA, data_type)
        audio_dir = os.path.join(args.audio_orig_path_AVA, data_type_map[data_type])

        # Ensure directories exist
        for video_id in df['video_id'].unique():
            dir_path = os.path.join(output_dir, video_id[0])
            if not os.path.isdir(dir_path):
                os.makedirs(dir_path)

        # Process each entity
        for entity in entity_list:
            entity_data = grouped_df.get_group(entity)
            video_key = entity_data.iloc[0]['video_id']
            start_time = entity_data.iloc[0]['frame_timestamp']
            end_time = entity_data.iloc[-1]['frame_timestamp']
            entity_id = entity_data.iloc[0]['entity_id']
            
            instance_path = os.path.join(output_dir, video_key, f'{entity_id}.wav')

            # Load and store audio data
            if video_key not in audio_features:
                audio_file = os.path.join(audio_dir, f'{video_key}.wav')
                sample_rate, audio_data = wavfile.read(audio_file)
                audio_features[video_key] = audio_data
            
            audio_start = int(float(start_time) * sample_rate)
            audio_end = int(float(end_time) * sample_rate)

            clip_audio_data = audio_features[video_key][audio_start:audio_end]
            wavfile.write(instance_path, sample_rate, clip_audio_data)


def extract_face_clips_from_videos(args):
    dataset_type_to_folder = {'train': 'trainval', 'val': 'trainval'}
    
    for data_type in ['train', 'val']:
        # Load CSV with annotations for the given dataset type
        annotations_df = pd.read_csv(os.path.join(args.trial_path_AVA, f'{data_type}_orig.csv'))
        negative_annotations = pd.concat([annotations_df[annotations_df['label_id'] == 0], annotations_df[annotations_df['label_id'] == 2]])
        positive_annotations = annotations_df[annotations_df['label_id'] == 1]
        combined_annotations = pd.concat([positive_annotations, negative_annotations]).reset_index(drop=True)
        
        combined_annotations_sorted = combined_annotations.sort_values(['entity_id', 'frame_timestamp']).reset_index(drop=True)
        
        unique_entities = combined_annotations_sorted['entity_id'].unique().tolist()
        annotations_grouped_by_entity = combined_annotations_sorted.groupby('entity_id')
        
        output_directory = os.path.join(args.visual_path_AVA, data_type)
        source_video_directory = os.path.join(args.visual_orig_path_AVA, dataset_type_to_folder[data_type])
        
        # Ensure an output directory exists for each video based on its ID
        for video_id in annotations_grouped_by_entity['video_id'].unique().tolist():
            video_output_dir = os.path.join(output_directory, video_id[0])
            if not os.path.isdir(video_output_dir):
                os.makedirs(video_output_dir)
        
        for entity in unique_entities:
            entity_annotations = annotations_grouped_by_entity.get_group(entity)
            video_id = entity_annotations.iloc[0]['video_id']
            entity_id = entity_annotations.iloc[0]['entity_id']

            video_file_path = glob.glob(os.path.join(source_video_directory, f'{video_id}.*'))[0]
            video_capture = cv2.VideoCapture(video_file_path)
            
            entity_output_dir = os.path.join(output_directory, video_id, entity_id)
            if not os.path.isdir(entity_output_dir):
                os.makedirs(entity_output_dir)
            
            # Process each frame annotated for the current entity
            for _, annotation_row in entity_annotations.iterrows():
                timestamp = annotation_row['frame_timestamp']
                image_filename = os.path.join(entity_output_dir, f"{timestamp:.2f}.jpg")
                video_capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
                success, frame = video_capture.read()
                
                if success:
                    frame_height, frame_width = frame.shape[:2]
                    x1 = int(annotation_row['entity_box_x1'] * frame_width)
                    y1 = int(annotation_row['entity_box_y1'] * frame_height)
                    x2 = int(annotation_row['entity_box_x2'] * frame_width)
                    y2 = int(annotation_row['entity_box_y2'] * frame_height)
                    cropped_face = frame[y1:y2, x1:x2]
                    
                    # Save the cropped face image
                    cv2.imwrite(image_filename, cropped_face)