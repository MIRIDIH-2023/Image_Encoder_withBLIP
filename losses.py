import torch.nn as nn
import torch

class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()
        """
        loss=0.00 -> cos_sim = 1
        loss=0.05 -> cos_sim = 0.9
        loss=0.10 -> cos_sim = 0.8
        loss=0.15 -> cos_sim = 0.7
        loss=0.20 -> cos_sim = 0.6
        loss=0.25 -> cos_sim = 0.5
        """    
    def forward(self, x1, x2):
        cos_sim = nn.functional.cosine_similarity(x1, x2) #같으면 1 다르면 -1
        cos_sim = ( cos_sim + 1 )/2 # 같으면 1 다르면 0

        return torch.mean(1 - cos_sim)
    