import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, time, numpy, os, subprocess, pandas, tqdm, re
from subprocess import PIPE

from loss import lossAV, lossV
from model.Model import ASD_Model

"""
Code inspired from https://github.com/Junhua-Liao/Light-ASD
"""

class ASD(nn.Module):
    def __init__(self, lr = 0.001, lrDecay = 0.95, l1_strength=0.01, **kwargs):
        super(ASD, self).__init__()
        self.model = ASD_Model(1024).cuda()
        self.lossAV = lossAV(1024).cuda()
        # self.optim = torch.optim.Adam(self.parameters(), lr = lr)
        self.optim = torch.optim.Adam(self.parameters(), lr = lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size = 1, gamma=lrDecay)
        # self.l1_strength = l1_strength
        print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%(sum(param.numel() for param in self.model.parameters()) / 1000 / 1000))

    def train_network(self, loader, epoch, r, **kwargs):
        self.train()
        self.scheduler.step(epoch - 1)  # StepLR
        index, top1, lossV, lossAV, loss = 0, 0, 0, 0, 0
        lr = self.optim.param_groups[0]['lr']
        for num, (features, labels) in enumerate(loader, start=1):
            self.zero_grad()
            # features_flat = features.view(-1, 1024).cuda()
            outsAV= self.model.forward_audio_visual_backend(features[0].cuda())  
            labels = labels[0].reshape((-1)).cuda() # Loss
            nlossAV, _, _, prec = self.lossAV.forward(outsAV, labels, r)
            # nlossAV, _, _, prec = self.lossAV.forward(features_flat, labels)
            # Calculate L1 regularization loss
            # l1_penalty = 0
            # for param in self.model.parameters():
            #     l1_penalty += self.l1_strength * torch.norm(param, 1)

            nloss = nlossAV
            lossAV += nlossAV.detach().cpu().numpy()
            loss += nloss.detach().cpu().numpy()
            top1 += prec
            nloss.backward()
            self.optim.step()
            index += len(labels)
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
            " [%2d] r: %2f, Lr: %5f, Training: %.2f%%, "    %(epoch, r, lr, 100 * (num / loader.__len__())) + \
            " LossV: %.5f, LossAV: %.5f, Loss: %.5f, ACC: %2.2f%% \r"  %(lossV/(num), lossAV/(num), loss/(num), 100 * (top1/index)))
            sys.stderr.flush()  

        sys.stdout.write("\n")      

        return loss/num, lr

    def evaluate_network(self, loader, evalCsvSave, evalOrig, **kwargs):
        self.eval()
        predScores = []
        # for audioFeature, visualFeature, labels in tqdm.tqdm(loader):
        for features, labels in tqdm.tqdm(loader):
            with torch.no_grad():
                # audioEmbed  = self.model.forward_audio_frontend(audioFeature[0].cuda())
                # visualEmbed = self.model.forward_visual_frontend(visualFeature[0].cuda())
                # features_flat = features.view(-1, 1024).cuda()
                outsAV= self.model.forward_audio_visual_backend(features[0].cuda())  
                labels = labels[0].reshape((-1)).cuda()             
                _, predScore, _, _ = self.lossAV.forward(outsAV, labels)
                # _, predScore, _, _ = self.lossAV.forward(features_flat, labels) 
                predScore = predScore[:,1].detach().cpu().numpy()
                predScores.extend(predScore)
                # break
        evalLines = open(evalOrig).read().splitlines()[1:]
        labels = []
        labels = pandas.Series( ['SPEAKING_AUDIBLE' for line in evalLines])
        scores = pandas.Series(predScores)
        evalRes = pandas.read_csv(evalOrig)
        evalRes['score'] = scores
        evalRes['label'] = labels
        evalRes.drop(['label_id'], axis=1,inplace=True)
        evalRes.drop(['instance_id'], axis=1,inplace=True)
        evalRes.to_csv(evalCsvSave, index=False)
        # cmd = "python -O utils/get_ava_active_speaker_performance.py -g %s -p %s "%(evalOrig, evalCsvSave)
        # mAP = float(str(subprocess.run(cmd, shell=True, stdout=PIPE, stderr=PIPE).stdout).split(' ')[2][:5])
        try:
            # Construct command
            cmd = f"python -O utils/get_ava_active_speaker_performance.py -g {evalOrig} -p {evalCsvSave}"
            
            # Execute command
            proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            # Use regex to find floating point numbers followed by a percentage sign
            match = re.search(r'(\d+\.\d+)%', proc.stdout)
            if match:
                mAP = float(match.group(1))
            else:
                raise ValueError("mAP value not found in the output.", proc.stderr)

        except Exception as e:
            print(f"Error parsing mAP from output: {e}")
            mAP = 0.0  # Default to 0.0 if there's an error
        return mAP

    

    def saveParameters(self, path):
        torch.save(self.state_dict(), path)

    def loadParameters(self, path):
        selfState = self.state_dict()
        loadedState = torch.load(path)
        for name, param in loadedState.items():
            origName = name;
            if name not in selfState:
                name = name.replace("module.", "")
                if name not in selfState:
                    print("%s is not in the model."%origName)
                    continue
            if selfState[name].size() != loadedState[origName].size():
                sys.stderr.write("Wrong parameter length: %s, model: %s, loaded: %s"%(origName, selfState[name].size(), loadedState[origName].size()))
                continue
            selfState[name].copy_(param)
