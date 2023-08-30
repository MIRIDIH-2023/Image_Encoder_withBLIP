import torch
import torch.nn as nn
import torch.optim as optim
from losses import CosineSimilarityLoss
from tqdm import tqdm


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def train(model, train_dataloader, valid_dataloader, logger, config):
    model.to(device)
    
    num_epochs= config['EPOCH']
    optimizer = optim.AdamW(model.parameters(), lr=config['LR'])
    criterion = CosineSimilarityLoss()
    
    
    logger_step = 0
    for epoch in range(num_epochs):
        ################################### one epoch training start ##################################################
        model.train()
        train_loss = 0.0
        for i, (text_vector, image_vector) in enumerate(tqdm(train_dataloader, desc=f"train epoch {epoch}", unit="batch")):
            text_vector = text_vector.to(device)
            image_vector = image_vector.to(device)  # 이미지 벡터도 디바이스로 이동

            optimizer.zero_grad()  # 매 반복마다 그래디언트 초기화

            latent_vector = model(image_vector)
            loss = criterion(text_vector, latent_vector)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            avg_loss = train_loss / (i + 1)
            tqdm.set_postfix({"avg_loss": str(avg_loss)}, refresh=True)  # tqdm에 평균 손실값 업데이트
            
            if(i%100==99):
                logger_step+=1
                logger.add_scalar("Training batch Loss", avg_loss , logger_step)  #logging train loss every 100 step
        
        logger.add_scalar("Training total Loss", avg_loss , epoch)  #logging train loss every 100 step
        ######################################## one epoch training end ##################################################
        
        ######################################## one epoch validation start ##################################################
        with torch.no_grad():
            model.eval()
            valid_loss = 0.0
            for i, (text_vector, image_vector) in enumerate(tqdm(valid_dataloader, desc=f"validatoin", unit="batch")):
                text_vector = text_vector.to(device)
                image_vector = image_vector.to(device)  # 이미지 벡터도 디바이스로 이동

                latent_vector = model(image_vector)
                loss = criterion(text_vector, latent_vector)

                valid_loss += loss.item()

                avg_loss = valid_loss / (i + 1)
                tqdm.set_postfix({"valid avg_loss": str(avg_loss)}, refresh=True)  # tqdm에 평균 손실값 업데이트
                
            logger.add_scalar("Validation total Loss", avg_loss , epoch)  #logging train loss every 100 step
        ######################################## one epoch validation end ##################################################
