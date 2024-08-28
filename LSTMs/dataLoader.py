import os, torch, numpy, cv2, random, glob, python_speech_features
from scipy.io import wavfile
from torchvision.transforms import RandomCrop
import pickle

def load_data(filepath):
    with open(filepath, 'rb') as file:
        data = pickle.load(file)
    return data

def load_features(data, picklePath, numFrames):
    entity_id = data[0]
    videoName = data[0][:11]
    pickleVideoFolderPath = os.path.join(picklePath, videoName)
    pickleEntityFolderPath = os.path.join(pickleVideoFolderPath, entity_id)
    pkl_files = sorted(os.listdir(pickleEntityFolderPath))
    frame_features = []
    for pkl_file in pkl_files[:numFrames]:
        filepath = os.path.join(pickleEntityFolderPath, pkl_file)
        features = load_data(filepath)
        frame_features.append(features)
    
    # Combine all arrays vertically (assuming they all have the same number of columns)
    if frame_features:
        frame_features = numpy.vstack(frame_features)
    return frame_features

def load_label(data, numFrames):
    res = []
    labels = data[3].replace('[', '').replace(']', '')
    labels = labels.split(',')
    for label in labels:
        res.append(int(label))
    res = numpy.array(res[:numFrames])
    return res

class train_loader(object):
    def __init__(self, trialFileName, audioPath, visualPath, batchSize, **kwargs):
        self.audioPath  = audioPath
        self.visualPath = visualPath
        self.miniBatch = []      
        mixLst = open(trialFileName).read().splitlines()
        # sort the training set by the length of the videos, shuffle them to make more videos in the same batch belong to different movies
        sortedMixLst = sorted(mixLst, key=lambda data: (int(data.split('\t')[1]), int(data.split('\t')[-1])), reverse=True)               
        start = 0       
        while True:
            length = int(sortedMixLst[start].split('\t')[1])
            end = min(len(sortedMixLst), start + max(int(batchSize / length), 1))
            self.miniBatch.append(sortedMixLst[start:end])
            if end == len(sortedMixLst):
                break
            start = end     

    def __getitem__(self, index):
        batchList    = self.miniBatch[index]
        numFrames   = int(batchList[-1].split('\t')[1])
        audioFeatures, visualFeatures, labels = [], [], []
        audioSet = generate_audio_set(self.audioPath, batchList) # load the audios in this batch to do augmentation
        for line in batchList:
            data = line.split('\t') 
            print(data[0])           
            audioFeatures.append(load_audio(data, self.audioPath, numFrames, audioAug = True, audioSet = audioSet))
            visualFeatures.append(load_visual(data, self.visualPath,numFrames, visualAug = True))
            labels.append(load_label(data, numFrames))
        return torch.FloatTensor(numpy.array(audioFeatures)), \
               torch.FloatTensor(numpy.array(visualFeatures)), \
               torch.LongTensor(numpy.array(labels))        

    def __len__(self):
        return len(self.miniBatch)


class val_loader(object):
    def __init__(self, trialFileName, audioPath, visualPath, **kwargs):
        self.audioPath  = audioPath
        self.visualPath = visualPath
        self.miniBatch = open(trialFileName).read().splitlines()

    def __getitem__(self, index):
        line       = [self.miniBatch[index]]
        numFrames  = int(line[0].split('\t')[1])
        audioSet   = generate_audio_set(self.audioPath, line)        
        data = line[0].split('\t')
        audioFeatures = [load_audio(data, self.audioPath, numFrames, audioAug = False, audioSet = audioSet)]
        visualFeatures = [load_visual(data, self.visualPath,numFrames, visualAug = False)]
        labels = [load_label(data, numFrames)]         
        return torch.FloatTensor(numpy.array(audioFeatures)), \
               torch.FloatTensor(numpy.array(visualFeatures)), \
               torch.LongTensor(numpy.array(labels))

    def __len__(self):
        return len(self.miniBatch)
    
class train_loader_res_18(object):
    def __init__(self, trialFileName, featurePath, batchSize, **kwargs):
        self.featurePath = featurePath
        self.miniBatch = []      
        mixLst = open(trialFileName).read().splitlines()
        # sort the training set by the length of the videos, shuffle them to make more videos in the same batch belong to different movies
        sortedMixLst = sorted(mixLst, key=lambda data: (int(data.split('\t')[1]), int(data.split('\t')[-1])), reverse=True)               
        start = 0       
        while True:
            length = int(sortedMixLst[start].split('\t')[1])
            end = min(len(sortedMixLst), start + max(int(batchSize / length), 1))
            self.miniBatch.append(sortedMixLst[start:end])
            if end == len(sortedMixLst):
                break
            start = end     

    def __getitem__(self, index):
        batchList    = self.miniBatch[index]
        numFrames   = int(batchList[-1].split('\t')[1])
        features, labels = [], []
        for line in batchList:
            data = line.split('\t')         
            features.append(load_features(data, self.featurePath , numFrames))
            labels.append(load_label(data, numFrames))
        return torch.FloatTensor(numpy.array(features)), \
               torch.LongTensor(numpy.array(labels))        

    def __len__(self):
        return len(self.miniBatch)
    
class val_loader_res_18(object):
    def __init__(self, trialFileName, featurePath, batchSize, **kwargs):
        self.featurePath = featurePath
        self.miniBatch = open(trialFileName).read().splitlines()

    def __getitem__(self, index):
        line       = [self.miniBatch[index]]
        numFrames  = int(line[0].split('\t')[1])    
        data = line[0].split('\t')
        features = [load_features(data, self.featurePath, numFrames)]
        labels = [load_label(data, numFrames)]
        return torch.FloatTensor(numpy.array(features)), \
               torch.LongTensor(numpy.array(labels))

    def __len__(self):
        return len(self.miniBatch)