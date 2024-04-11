import os
import tarfile
import requests
from tempfile import TemporaryDirectory
import shutil 

def download_file(url, local_path):
    """Download a file from a URL to a local path."""
    response = requests.get(url, stream=True)
    with open(local_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                file.write(chunk)

def extract_tarfile(tar_path, extract_path):
    """Extract a tar file to a given path."""
    with tarfile.open(tar_path, 'r:*') as tar:
        tar.extractall(path=extract_path)

def download_and_extract(url, target_dir):
    """Download and extract tar.bz2 files from a URL into a target directory."""
    with TemporaryDirectory() as temp_dir:
        # Download the file
        filename = os.path.join(temp_dir, os.path.basename(url))
        print(f"Downloading {url}...")
        download_file(url, filename)
        
        # Extract the downloaded file
        print(f"Extracting {filename}...")
        extract_tarfile(filename, temp_dir)
        
        # Find the folder inside the extracted directory
        nested_folders = [f.path for f in os.scandir(temp_dir) if f.is_dir()]
        for folder in nested_folders:
            # Extract contents of the nested folder into the target directory
            for item in os.listdir(folder):
                item_path = os.path.join(folder, item)
                if os.path.isfile(item_path):
                    os.makedirs(target_dir, exist_ok=True)
                    shutil.move(item_path, os.path.join(target_dir, item))  # Use shutil.move here

def download_video_file(file_name_without_extension, extensions, data_set_video_urls, save_folder, BASE_URL, mock=False):
    downloaded = False
    for data_set_url in data_set_video_urls:
        video_type_folder = data_set_url.strip('/')
        local_save_folder = os.path.join(save_folder, video_type_folder)
        os.makedirs(local_save_folder, exist_ok=True)
        for ext in extensions:
            file_name = f"{file_name_without_extension}.{ext}"
            url = f"{BASE_URL}{data_set_url}{file_name}"
            if not mock:
                try:
                    r = requests.get(url, stream=True)
                    if r.status_code == 200:
                        total_size = int(r.headers.get('content-length', 0))
                        downloaded_size = 0
                        path = os.path.join(local_save_folder, file_name)
                        with open(path, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)
                                downloaded_size += len(chunk)
                                done = int(50 * downloaded_size / total_size)
                                print(f"\rDownloading {file_name}: [{'=' * done}{' ' * (50-done)}] {downloaded_size/total_size:.2%}", end='')
                        print()  # Ensure the next print is on a new line
                        downloaded = True
                        break  # Stop trying other extensions once download is successful
                except Exception as e:
                    pass  # Optionally handle errors or log them silently
            else:
                downloaded = True
                break  # Assume mock download successful, stop trying other extensions
    if not downloaded:
        pass  # Optionally handle the failure to download any files
def download_dataset_videos():
    # URLs of the dataset files
    train_url = "https://s3.amazonaws.com/ava-dataset/annotations/ava_activespeaker_train_v1.0.tar.bz2"
    val_url = "https://s3.amazonaws.com/ava-dataset/annotations/ava_activespeaker_val_v1.0.tar.bz2"

    # Target directories
    train_dir = "audio_video_csvs_train"
    val_dir = "audio_video_csvs_val"

    # Create directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Download and extract datasets
    download_and_extract(train_url, train_dir)
    download_and_extract(val_url, val_dir)


    BASE_URL = 'https://s3.amazonaws.com/ava-dataset/'
    folder_names = ['audio_video_csvs_train', 'audio_video_csvs_val']

    save_folder = 'ava_dataset'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)


    file_names = os.listdir(save_folder)
    existing_video_names = [os.path.splitext(name)[0] for name in file_names]
    video_names_to_download = []

    for folder_name in folder_names:
        try:
            files = os.listdir(folder_name)
            for file_name in files:
                video_name = file_name.rsplit('-', 1)[0]
                if video_name not in existing_video_names:
                    video_names_to_download.append(video_name)
        except FileNotFoundError:
            print(f"Warning: Folder {folder_name} not found.")

    video_names_to_download = sorted(set(video_names_to_download))

    extensions = ["mkv", "mp4", "webm"]
    data_set_video_urls = ['trainval/', 'test/']  # Adjust according to actual dataset structure

    for video_name in video_names_to_download:
        download_video_file(video_name, extensions, data_set_video_urls, save_folder, BASE_URL, mock=False)

    # Let the user know the script has finished
    print('All files downloaded.')
