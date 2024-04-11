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



# This preprocess is based on source [2]
def preprocess_AVA(args):
    if not args:
        args = argparse.Namespace()
        args = init_args()
    extract_audio(args) # Take 1 hour
    extract_audio_clips(args) # Takes 3 minutes
    extract_face_clips_from_videos(args) # Takes about 1 days


def extract_audio(args):
    # Takes 1 hour to extract the audio from videos
    for data_type in ['train', 'val']:
        inpFolder = '%s/%s'%(args.visual_orig_path_AVA, data_type)
        outFolder = '%s/%s'%(args.audio_orig_path_AVA, data_type)
        os.makedirs(outFolder, exist_ok = True)
        videos = glob.glob("%s/*"%(inpFolder))
        for videoPath in videos:
            audioPath = '%s/%s'%(outFolder, videoPath.split('/')[-1].split('.')[0] + '.wav')
            cmd = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads 8 %s -loglevel panic" % (videoPath, audioPath))
            subprocess.call(cmd, shell=True, stdout=None)


def extract_audio_clips(args):
    # Take 3 minutes to extract the audio clips
    dic = {'train':'train', 'val':'val'}
    for data_type in ['train', 'val']:
        df = pd.read_csv(os.path.join(args.trial_path_AVA, f'combined_{data_type}.csv'), engine='python')
        # Separate data into negative and positive labels (ignoring label '2' for negative)
        dfNeg = pd.concat([df[df['label_id'] == 0], df[df['label_id'] == 2]])
        dfPos = df[df['label_id'] == 1]
        # Combine positive and negative dataframes and reset index
        df = pd.concat([dfPos, dfNeg]).reset_index(drop=True)
        df = df.sort_values(['entity_id', 'frame_timestamp']).reset_index(drop=True)

        # Get unique entity IDs
        entity_list = df['entity_id'].unique().tolist()
        df = df.groupby('entity_id')
        audioFeatures = {}

        outDir = os.path.join(args.audio_path_AVA, data_type)
        audioDir = os.path.join(args.audio_orig_path_AVA, dic[data_type])

        # Ensure output directories exist for each video ID
        for l in df['video_id'].unique().tolist():
            d = os.path.join(outDir, l[0])
            if not os.path.isdir(d):
                os.makedirs(d)

        # Process each entity in the dataset
        for entity in entity_list:
            ins_data = df.get_group(entity)
            videoKey = ins_data.iloc[0]['video_id']
            start = ins_data.iloc[0]['frame_timestamp']
            end = ins_data.iloc[-1]['frame_timestamp']
            entityID = ins_data.iloc[0]['entity_id']

            insPath = os.path.join(outDir, videoKey, f'{entityID}.wav')

            # If the audio feature for the video hasn't been loaded, load and store it
            if videoKey not in audioFeatures.keys():                
                audioFile = os.path.join(audioDir, f'{videoKey}.wav')
                sr, audio = wavfile.read(audioFile)
                audioFeatures[videoKey] = audio
            
            audioStart = int(float(start) * sr)
            audioEnd = int(float(end) * sr)

            audioData = audioFeatures[videoKey][audioStart:audioEnd]
            wavfile.write(insPath, sr, audioData)


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