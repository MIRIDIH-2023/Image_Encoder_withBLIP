import torch.nn as nn
import torch

class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()
        
    def forward(self, x1, x2):
        cos_sim = nn.functional.cosine_similarity(x1, x2) #같으면 1 다르면 -1
        cos_sim = ( cos_sim + 1 )/2 # 같으면 1 다르면 0

        return torch.mean(1 - cos_sim)