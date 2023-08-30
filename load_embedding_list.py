import torch
import pickle
import numpy as np

def loading():
    #keyword, text, caption embedding list의 dtype = tensor
    keyword_embedding_list = torch.load('keyword_embedding_list.pth')
    text_embedding_list = torch.load('text_embedding_list.pth')
    caption_embedding_list = torch.load('caption_embedding_list.pth')

    #image embedding list의 dtype = numpy. dataset class에서 변환
    image_embedding_path = '/content/drive/MyDrive/ColabNotebooks/image_embedding_list.pickle'
    with open(image_embedding_path, 'rb') as f:
        image_embedding_list = pickle.load(f)
        
    return keyword_embedding_list, text_embedding_list, caption_embedding_list, image_embedding_list