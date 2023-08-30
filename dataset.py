from torch.utils.data import Dataset
import torch

class customDataset(Dataset):
    def __init__(self, text_vector_list , caption_vector_list, keyword_vector_list , image_vector_list, use_only_keyword=True):
        super(customDataset,self).__init__()
        
        self.use_only_keyword = use_only_keyword
        self.text_vector_list = text_vector_list
        self.caption_vector_list = caption_vector_list
        self.keyword_vector_list = keyword_vector_list
        self.image_vector_list = image_vector_list
        
        a = len(text_vector_list)
        b = len(caption_vector_list)
        c = len(keyword_vector_list)
        d = len(image_vector_list)
        assert a==b and b==c
        
        self.data_len = len(image_vector_list)


    def __len__(self):
        if self.use_only_keyword:
          return self.data_len
        else:
          return self.data_len * 3
        
    def __getitem__(self, idx):

        vector_pick = idx//self.data_len #determine which vector list to pick
        idx = idx%self.data_len #determine modified idx  range=[0,self.data_len)
        
        if self.use_only_keyword:
              text_embedded_list = self.keyword_vector_list
        else:
          if vector_pick==0:
              text_embedded_list = self.text_vector_list
          elif vector_pick==1:
              text_embedded_list = self.caption_vector_list
          else:
              text_embedded_list = self.keyword_vector_list
        
        selected_data = text_embedded_list[idx] #dtype = torch
        image_vector = self.image_vector_list[idx] #dypte = numpy
        image_vector = torch.tensor(image_vector)
        
        return selected_data , image_vector
  
    