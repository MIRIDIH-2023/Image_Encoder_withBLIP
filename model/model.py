import transformers
import torch
import torch.nn as nn
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class SelfAttention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, output_dim)
        self.key = nn.Linear(input_dim, output_dim)
        self.value = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_weights = self.softmax(attention_scores)
        output = torch.matmul(attention_weights, value)
        return output

class FeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class Image_Embedding_model_with_BLIP_imageEncoder(nn.Module):
    """
    image(jpg)를 input으로 받는 경우에 사용하는 모델
    embedded image vector를 input으로 받는 경우는 다른 class 사용
    """
    def __init__(self , num_blocks=2 ):
        super(Image_Embedding_model_with_BLIP_imageEncoder,self).__init__()
        
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        self.image_encoder = model.vision_model
        
        layers = []
        for _ in range(num_blocks):
            layers.append(SelfAttention(768, 768))
            layers.append(FeedForward(768, 512, drop_prob=0.0))
        self.attention_block = nn.Sequential(*layers)

        self.fc_layer = nn.Linear(768,512)
        
        self.freeze_layer()
        
    def forward(self, image):
        #image == 384 384 shape
        image = self.processor(image, return_tensors="pt").to(device)
        image_vector = self.image_encoder(image['pixel_values']).pooler_output # [batch , 577, 1024] -> [batch, 1024]
        output = self.attention_block(image_vector)
        output = self.fc_layer(output)
        
        return output

    def freeze_layer(self): #freeze image encoder
        for param in self.image_encoder.parameters():
            param.requires_grad = False



class Image_Embedding_model_only_Attention_Block(nn.Module):
    """
    embedded image vector를 input으로 받는 경우에 사용하는 모델
    image(jpg) 를 input으로 받는 경우는 다른 class 사용
    """
    def __init__(self , num_blocks=4 ):
        super(Image_Embedding_model_only_Attention_Block,self).__init__()
        
        layers = []
        for _ in range(num_blocks):
            layers.append(SelfAttention(768, 768))
            layers.append(FeedForward(768, 512, drop_prob=0.0))
        self.attention_block = nn.Sequential(*layers)

        self.fc_layer = nn.Linear(768,512)
        
        
    def forward(self, image_vector):
        """
        input : [# of batch, 768]
        """
        output = self.attention_block(image_vector)
        output = self.fc_layer(output)
        
        return output