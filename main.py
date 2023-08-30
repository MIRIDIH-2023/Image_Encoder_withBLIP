from load_embedding_list import loading
from train import train
from model.model import Image_Embedding_model_only_Attention_Block
from dataset import customDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from saving_image_vector import save_vector
import torch

from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def start(config):
    keyword_embedding_list , text_embedding_list , caption_embedding_list , image_embedding_list = loading()

    max_len = len(image_embedding_list)
    keyword_embedding_list = keyword_embedding_list[:max_len]
    text_embedding_list = text_embedding_list[:max_len]
    caption_embedding_list = caption_embedding_list[:max_len]
    

    log_dir = config["LOG_DIR"]  # 로그가 저장될 디렉토리 경로
    logger = SummaryWriter(log_dir)
    model = Image_Embedding_model_only_Attention_Block(num_blocks=config['NUM_ATTENTION_BLOCK']).to(device)

    keyword_train, keyword_valid, text_train, text_valid, caption_train, caption_valid, image_train, image_valid = train_test_split(
        keyword_embedding_list, text_embedding_list, 
        caption_embedding_list, image_embedding_list, 
        test_size=0.1, random_state=42, shuffle=True
    )

    train_dataset = customDataset(text_train,
                                caption_train,
                                keyword_train,
                                image_train,
                                use_only_keyword=True,
                                )

    valid_dataset = customDataset(text_valid,
                                caption_valid,
                                keyword_valid,
                                image_valid,
                                use_only_keyword=True,
                                )

    test_dataset = customDataset(text_embedding_list,
                                 caption_embedding_list,
                                 keyword_embedding_list,
                                 image_embedding_list,
                                 use_only_keyword=True)

    train_dataloader = DataLoader(train_dataset , batch_size=config['BATCH_SIZE'], shuffle=True)
    valid_dataloader = DataLoader(valid_dataset , batch_size=config['BATCH_SIZE'], shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=config['BATCH_SIZE'], shuffle=False)

    train(model,train_dataloader,valid_dataloader,logger, config)

    save_vector(model,test_dataloader,config)


if __name__=="__main__":
    config = {
        'LOG_DIR' : 'logs',
        'BATCH_SIZE' : 64,
        'EPOCH' : 5,
        'LR' : 0.001,
        'NUM_ATTENTION_BLOCK' : 4
    }
    
    start(config)