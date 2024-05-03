import os, torch, cv2, random, glob, python_speech_features
from scipy.io import wavfile
import numpy as np

def generate_audio_set(dataPath, batchList):
    audioSet = {}
    for line in batchList:
        data = line.split('\t')
        videoName = data[0][:11]
        dataName = data[0]
        _, audio = wavfile.read(os.path.join(dataPath, videoName, dataName + '.wav'))
        audioSet[dataName] = audio
    return audioSet

def overlap(dataName, audio, audioSet):   
    noiseName =  random.sample(set(list(audioSet.keys())) - {dataName}, 1)[0]
    noiseAudio = audioSet[noiseName]    
    snr = [random.uniform(-5, 5)]
    if len(noiseAudio) < len(audio):
        shortage = len(audio) - len(noiseAudio)
        noiseAudio = np.pad(noiseAudio, (0, shortage), 'wrap')
    else:
        noiseAudio = noiseAudio[:len(audio)]
    noiseDB = 10 * np.log10(np.mean(abs(noiseAudio ** 2)) + 1e-4)
    cleanDB = 10 * np.log10(np.mean(abs(audio ** 2)) + 1e-4)
    noiseAudio = np.sqrt(10 ** ((cleanDB - noiseDB - snr) / 10)) * noiseAudio
    audio = audio + noiseAudio    
    return audio.astype(np.int16)

def load_audio(data, dataPath, num_frames, audio_aug, audioSet = None):
    dataName = data[0]
    fps = float(data[2])    
    audio = audioSet[dataName]    
    if audio_aug == True:
        augType = random.randint(0,1)
        if augType == 1:
            audio = overlap(dataName, audio, audioSet)
        else:
            audio = audio
    # fps is not always 25, in order to align the visual, we modify the window and step in MFCC extraction process based on fps
    audio = python_speech_features.mfcc(audio, 16000, numcep = 13, winlen = 0.025 * 25 / fps, winstep = 0.010 * 25 / fps)
    maxAudio = int(num_frames * 4)
    if audio.shape[0] < maxAudio:
        shortage    = maxAudio - audio.shape[0]
        audio     = np.pad(audio, ((0, shortage), (0,0)), 'wrap')
    audio = audio[:int(round(num_frames * 4)),:]  
    return audio

def load_visual(data, dataPath, num_frames, visual_aug): 
    dataName = data[0]
    videoName = data[0][:11]
    faceFolderPath = os.path.join(dataPath, videoName, dataName)
    faceFiles = glob.glob("%s/*.jpg"%faceFolderPath)
    sortedFaceFiles = sorted(faceFiles, key=lambda data: (float(data.split('/')[-1][:-4])), reverse=False) 
    faces = []
    H = 112
    if visual_aug == True:
        new = int(H*random.uniform(0.7, 1))
        x, y = np.random.randint(0, H - new), np.random.randint(0, H - new)
        M = cv2.getRotationMatrix2D((H/2,H/2), random.uniform(-15, 15), 1)
        augType = random.choice(['orig', 'flip', 'crop', 'rotate']) 
    else:
        augType = 'orig'
    for faceFile in sortedFaceFiles[:num_frames]:
        face = cv2.imread(faceFile)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (H,H))
        if augType == 'orig':
            faces.append(face)
        elif augType == 'flip':
            faces.append(cv2.flip(face, 1))
        elif augType == 'crop':
            faces.append(cv2.resize(face[y:y+new, x:x+new] , (H,H))) 
        elif augType == 'rotate':
            faces.append(cv2.warpAffine(face, M, (H,H)))
    faces = np.array(faces)
    return faces

def load_label(data, num_frames):
    res = []
    labels = data[3].replace('[', '').replace(']', '')
    labels = labels.split(',')
    for label in labels:
        res.append(int(label))
    res = np.array(res[:num_frames])
    return res

class TrainLoader(object):
    def __init__(self, trial_file_name, audio_path, visual_path, batch_size, current_fold_video_ids=None, **kwargs):
        self.audio_path = audio_path
        self.visual_path = visual_path
        self.mini_batch = []
        mixLst = open(trial_file_name).read().splitlines()
        self.current_fold_video_ids = current_fold_video_ids
        if current_fold_video_ids is not None:
            # Extract the video_id from each line and filter based on current_fold_video_ids
            mixLst = [line for line in mixLst if line.rsplit('_', 2)[0] in current_fold_video_ids]

        # Rest of your code for sorting and creating mini batches
        sortedMixLst = sorted(mixLst, key=lambda data: (int(data.split('\t')[1]), random.random()), reverse=True)
        start = 0
        while True:
            length = int(sortedMixLst[start].split('\t')[1])
            end = min(len(sortedMixLst), start + max(int(batch_size / length), 1))
            self.mini_batch.append(sortedMixLst[start:end])
            if end == len(sortedMixLst):
                break
            start = end

        self.audio_set = generate_audio_set(self.audio_path, mixLst)
        

    def __getitem__(self, index):
        batchList = self.mini_batch[index]
        audio_set = {data.split('\t')[0]: self.audio_set[data.split('\t')[0]] 
                for data in batchList if data.split('\t')[0] in self.audio_set}
        num_frames   = int(batchList[-1].split('\t')[1])
        audio_features, visual_features, labels = [], [], []

        for line in batchList:
            data = line.split('\t')
            audio_features.append(load_audio(data, self.audio_path, num_frames, audio_aug = True, audioSet = audio_set))
            visual_features.append(load_visual(data, self.visual_path,num_frames, visual_aug = True))
            labels.append(load_label(data, num_frames))

        audio_tensor = torch.FloatTensor(np.array(audio_features))
        visual_tensor = torch.FloatTensor(np.array(visual_features))
        labels_tensor = torch.LongTensor(np.array(labels))
        return audio_tensor, visual_tensor, labels_tensor


    def __len__(self):
        return len(self.mini_batch)

class TrainLoaderAudio(object):
    def __init__(self, trial_file_name, audio_path, batch_size, current_fold_video_ids=None, **kwargs):
        self.audio_path = audio_path
        self.mini_batch = []
        mixLst = open(trial_file_name).read().splitlines()
        self.current_fold_video_ids = current_fold_video_ids
        
        if current_fold_video_ids is not None:
            # Extract the video_id from each line and filter based on current_fold_video_ids
            mixLst = [line for line in mixLst if line.rsplit('_', 2)[0] in current_fold_video_ids]

        # Rest of your code for sorting and creating mini batches
        sortedMixLst = sorted(mixLst, key=lambda data: (int(data.split('\t')[1]), random.random()), reverse=True)
        start = 0
        while True:
            length = int(sortedMixLst[start].split('\t')[1])
            end = min(len(sortedMixLst), start + max(int(batch_size / length), 1))
            self.mini_batch.append(sortedMixLst[start:end])
            if end == len(sortedMixLst):
                break
            start = end
        
        self.audio_set = generate_audio_set(self.audio_path, mixLst)

    def __getitem__(self, index):
        batchList = self.mini_batch[index]
        audio_set = {data.split('\t')[0]: self.audio_set[data.split('\t')[0]] 
                for data in batchList if data.split('\t')[0] in self.audio_set}
        num_frames   = int(batchList[-1].split('\t')[1])
        audio_features, labels = [], []

        for line in batchList:
            data = line.split('\t')
            audio_features.append(load_audio(data, self.audio_path, num_frames, audio_aug = True, audioSet = audio_set))
            labels.append(load_label(data, num_frames))

        audio_tensor = torch.FloatTensor(np.array(audio_features))
        visual_tensor = []
        labels_tensor = torch.LongTensor(np.array(labels))
        return audio_tensor, visual_tensor, labels_tensor  

    def __len__(self):
        return len(self.mini_batch)
    
class TrainLoaderVideo(object):
    def __init__(self, trial_file_name, visual_path, batch_size, current_fold_video_ids=None, **kwargs):
        self.visual_path = visual_path
        self.mini_batch = []
        mixLst = open(trial_file_name).read().splitlines()
        self.current_fold_video_ids = current_fold_video_ids
        
        if current_fold_video_ids is not None:
            # Extract the video_id from each line and filter based on current_fold_video_ids
            mixLst = [line for line in mixLst if line.rsplit('_', 2)[0] in current_fold_video_ids]

        # Rest of your code for sorting and creating mini batches
        sortedMixLst = sorted(mixLst, key=lambda data: (int(data.split('\t')[1]), random.random()), reverse=True)
        start = 0
        while True:
            length = int(sortedMixLst[start].split('\t')[1])
            end = min(len(sortedMixLst), start + max(int(batch_size / length), 1))
            self.mini_batch.append(sortedMixLst[start:end])
            if end == len(sortedMixLst):
                break
            start = end

    def __getitem__(self, index):
        batchList = self.mini_batch[index]
        num_frames   = int(batchList[-1].split('\t')[1])
        visual_features, labels = [], []

        for line in batchList:
            data = line.split('\t')
            visual_features.append(load_visual(data, self.visual_path,num_frames, visual_aug = True))
            labels.append(load_label(data, num_frames))

        audio_tensor = []
        visual_tensor = torch.FloatTensor(np.array(visual_features))
        labels_tensor = torch.LongTensor(np.array(labels))
        return audio_tensor, visual_tensor, labels_tensor  

    def __len__(self):
        return len(self.mini_batch)
    
class ValLoader(object):
    def __init__(self, trial_file_name, audio_path, visual_path, current_fold_video_ids=None, **kwargs):
        self.audio_path  = audio_path
        self.visual_path = visual_path
        self.current_fold_video_ids = current_fold_video_ids

        mixLst = open(trial_file_name).read().splitlines()
        if current_fold_video_ids is not None:
            # Extract the video_id from each line and filter based on current_fold_video_ids
            mixLst = [line for line in mixLst if line.rsplit('_', 2)[0] in current_fold_video_ids]
        
        self.mini_batch = mixLst
        self.audio_set = generate_audio_set(self.audio_path, mixLst)

    def __getitem__(self, index):
        line = [self.mini_batch[index]]
        num_frames = int(line[0].split('\t')[1])
        data = line[0].split('\t')
        video_id = data[0].rsplit('_', 2)[0]
        audio_set = {data[0]: self.audio_set[data[0]]}
        audio_features = [load_audio(data, self.audio_path, num_frames, audio_aug = False, audioSet = audio_set)]
        visual_features = [load_visual(data, self.visual_path,num_frames, visual_aug = False)]
        labels = [load_label(data, num_frames)]         
        return video_id, torch.FloatTensor(np.array(audio_features)), \
               torch.FloatTensor(np.array(visual_features)), \
               torch.LongTensor(np.array(labels))

    def __len__(self):
        return len(self.mini_batch)

class ValLoaderAudio(object):
    def __init__(self, trial_file_name, audio_path, current_fold_video_ids=None, **kwargs):
        self.audio_path  = audio_path
        self.current_fold_video_ids = current_fold_video_ids
        mixLst = open(trial_file_name).read().splitlines()
        
        if current_fold_video_ids is not None:
            # Extract the video_id from each line and filter based on current_fold_video_ids
            mixLst = [line for line in mixLst if line.rsplit('_', 2)[0] in current_fold_video_ids]
        
        self.mini_batch = mixLst
        self.audio_set = generate_audio_set(self.audio_path, mixLst)

    
    def __getitem__(self, index):
        line    = [self.mini_batch[index]]
        num_frames  = int(line[0].split('\t')[1])
        data = line[0].split('\t')
        video_id = data[0].rsplit('_', 2)[0]
        audio_set = {data[0]: self.audio_set[data[0]]}
        audio_features = [load_audio(data, self.audio_path, num_frames, audio_aug = False, audioSet = audio_set)]
        labels = [load_label(data, num_frames)]         
        return video_id, torch.FloatTensor(np.array(audio_features)), \
               [], torch.LongTensor(np.array(labels))

    def __len__(self):
        return len(self.mini_batch)

class ValLoaderVideo(object):
    def __init__(self, trial_file_name, visual_path, current_fold_video_ids=None, **kwargs):
        self.visual_path = visual_path
        self.current_fold_video_ids = current_fold_video_ids
        mixLst = open(trial_file_name).read().splitlines()
        
        if current_fold_video_ids is not None:
            # Extract the video_id from each line and filter based on current_fold_video_ids
            mixLst = [line for line in mixLst if line.rsplit('_', 2)[0] in current_fold_video_ids]
        self.mini_batch = mixLst


    def __getitem__(self, index):
        line       = [self.mini_batch[index]]
        num_frames  = int(line[0].split('\t')[1])
        data = line[0].split('\t')
        video_id = data[0].rsplit('_', 2)[0]
        visual_features = [load_visual(data, self.visual_path,num_frames, visual_aug = False)]
        labels = [load_label(data, num_frames)]         
        return video_id, [], \
               torch.FloatTensor(np.array(visual_features)), \
               torch.LongTensor(np.array(labels))

    def __len__(self):
        return len(self.mini_batch)
    
