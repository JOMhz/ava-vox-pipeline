import torch
import torch.nn as nn
import torch.nn.functional as F

# Loss code built on from work done by source [3]
class lossAV(nn.Module):
	def __init__(self):
		super(lossAV, self).__init__()
		self.criterion = nn.CrossEntropyLoss()
		self.FC        = nn.Linear(256, 2)
	

	def forward(self, x, labels=None):
		x = x.squeeze(1)
		x = self.FC(x)
		if labels is None:
			predScore = F.softmax(x, dim=-1)[:, 1]
			return predScore.detach().cpu().numpy()
		else:
			nloss = self.criterion(x, labels)
			_, predLabel = torch.max(F.softmax(x, dim=-1), dim=1)
			correctNum = (predLabel == labels).sum().float()
			return nloss, F.softmax(x, dim=-1), predLabel, correctNum

class lossA(nn.Module):
    def __init__(self):
        super(lossA, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.FC = nn.Linear(128, 2)

    def forward(self, x, labels=None):
        x = x.squeeze(1)
        x = self.FC(x)
        if labels is None:
            predScore = F.softmax(x, dim=-1)[:, 1]
            return predScore.detach().cpu().numpy()
        else:
            nloss = self.criterion(x, labels)
            _, predLabel = torch.max(F.softmax(x, dim=-1), dim=1)
            correctNum = (predLabel == labels).sum().float()
            return nloss, F.softmax(x, dim=-1), predLabel, correctNum

class lossV(nn.Module):
    def __init__(self):
        super(lossV, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.FC = nn.Linear(128, 2)

    def forward(self, x, labels=None):
        x = x.squeeze(1)
        x = self.FC(x)
        if labels is None:
            predScore = F.softmax(x, dim=-1)[:, 1]
            return predScore.detach().cpu().numpy()
        else:
            nloss = self.criterion(x, labels)
            _, predLabel = torch.max(F.softmax(x, dim=-1), dim=1)
            correctNum = (predLabel == labels).sum().float()
            return nloss, F.softmax(x, dim=-1), predLabel, correctNum
