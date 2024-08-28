import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Code from https://github.com/Junhua-Liao/Light-ASD
"""

# class lossAV(nn.Module):
# 	def __init__(self, channels):
# 		super(lossAV, self).__init__()
# 		self.criterion = nn.BCEWithLogitsLoss()
# 		self.FC = nn.Linear(channels, 2)

# 	def forward(self, x, labels=None):
# 		x = x.squeeze(1)
# 		logits = self.FC(x)
# 		if labels is None:
# 			scores = torch.sigmoid(logits)[:, 1]  # Probability for class 1
# 			return scores.detach().cpu().numpy()
# 		else:
# 			labels_one_hot = F.one_hot(labels, num_classes=2).float()  # Convert scalar to one-hot
# 			print(labels_one_hot)
# 			F.softmax(x1, dim = -1)[:,1].float()
# 			loss = self.criterion(logits, labels_one_hot)
# 			# loss = self.criterion(logits, labels.unsqueeze(1).float())
# 			pred_scores = torch.sigmoid(logits)
# 			pred_labels = (pred_scores > 0.5).float()[:, 1]  # Binary predictions
# 			correct_num = (pred_labels == labels).sum().float()
# 			return loss, pred_scores, pred_labels, correct_num

			
class lossAV(nn.Module):
	def __init__(self, channels):
		super(lossAV, self).__init__()
		self.criterion = nn.BCELoss()
		self.FC        = nn.Linear(channels, 2)

	def forward(self, x, labels = None, r = 1):	
		x = x.squeeze(1)
		x = self.FC(x)
		if labels == None:
			predScore = x[:,1]
			predScore = predScore.t()
			predScore = predScore.view(-1).detach().cpu().numpy()
			return predScore
		else:
			x1 = x / r
			x1 = F.softmax(x1, dim = -1)[:,1]
			nloss = self.criterion(x1, labels.float())
			predScore = F.softmax(x, dim = -1)
			predLabel = torch.round(F.softmax(x, dim = -1))[:,1]
			correctNum = (predLabel == labels).sum().float()
			return nloss, predScore, predLabel, correctNum




class lossV(nn.Module):
	def __init__(self):
		super(lossV, self).__init__()
		self.criterion = nn.BCELoss()
		self.FC        = nn.Linear(128, 2)

	def forward(self, x, labels, r = 1):	
		x = x.squeeze(1)
		x = self.FC(x)
		
		x = x / r
		x = F.softmax(x, dim = -1)

		nloss = self.criterion(x[:,1], labels.float())
		return nloss