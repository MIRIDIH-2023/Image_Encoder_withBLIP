import torch
from tqdm import tqdm
import numpy as np

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def save_vector(model, dataloder, config):
    model.load_state_dict(torch.load('image_embedding_model.pth'))
    model.eval()
    model.to(device)
    print('saving vector...')
    image_latent_vector_list = []
    for _ , image_vector in tqdm(dataloder):
        image_vector = image_vector.to(device)  # 이미지 벡터도 디바이스로 이동
        latent_vector = model(image_vector) #[batch , 512]
        
        image_latent_vector_list.extend(latent_vector.cpu().detach().numpy())
    print(len(image_latent_vector_list))
    for i in range(len(image_latent_vector_list)):
        image_latent_vector_list[i] = torch.Tensor(image_latent_vector_list[i])
    torch.save(image_latent_vector_list, 'image_latent_vector_list.pth')
    print('done!')
    print("save image latent vector at image_latent_vector_list.pth")

"""
pickle_file_path = '/content/drive/MyDrive/ColabNotebooks/image_embedding_list.pickle'
image_embedding_list = []

for idx in tqdm(range(14000,len(image_path_list))):
    image_path = image_path_list[idx]
    image = Image.open(BytesIO(requests.get(image_path).content))
    image = processor(image, return_tensors="pt")['pixel_values']
    image = image.to('cuda')

    image_vector = image_encoder(image).pooler_output[0] # [1 , 577, 768] -> [768]

    image_embedding_list.extend([image_vector.cpu().detach().numpy()])

    if(idx%1000==999):
      if os.path.exists(pickle_file_path):
        with open(pickle_file_path, 'rb') as f:
            existing_data = pickle.load(f)

        new_list = existing_data + image_embedding_list

        with open(pickle_file_path, 'wb') as f:
            pickle.dump(new_list, f)

      else:
          with open(pickle_file_path, 'wb') as f:
            pickle.dump(image_embedding_list, f)
      image_embedding_list = []


with open(pickle_file_path, 'rb') as f:
  existing_data = pickle.load(f)

new_list = existing_data + image_embedding_list

with open(pickle_file_path, 'wb') as f:
  pickle.dump(new_list, f)
"""