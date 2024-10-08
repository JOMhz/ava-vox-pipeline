import subprocess
import glob
import os

"""
    This code is by https://github.com/fuankarion/active-speakers-context/tree/master
"""

def extractAudioFromVideo(video, targetAudio):
    command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % (video, targetAudio))
    subprocess.call(command, shell=True, stdout=None)

if __name__ == '__main__':
    ava_video_dir = '.../AVA/videos/trainval'
    target_audios = '.../audio_tracks/trainval'

    all_videos = os.listdir(ava_video_dir)
    all_videos = [v.split('.')[0] for v in all_videos]

    for video_name in all_videos:
        print('process video ', video_name)
        actual_file_name = glob.glob(os.path.join(ava_video_dir, video_name+'*'))
        print(actual_file_name)
        extractAudioFromVideo(actual_file_name[0], os.path.join(target_audios,
                              video_name+'.wav'))
