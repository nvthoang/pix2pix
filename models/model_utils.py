import numpy as np
import torchvision.transforms as T
import torch
import cv2


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class generate_img():
    def __init__(self, generator, resize_to:tuple=(1024,1024), 
                 threshold=0.5, max_value=1, cuda:bool=True): 
        self.resize_to=resize_to
        self.model=generator
        self.threshold=threshold
        self.max_value=max_value
        if cuda: 
            self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device='cpu'

    def gen2binary(self, generated_img, threshold=0.5, max_value=1):
        img = generated_img
        _, thresh = cv2.threshold(img, threshold, max_value, cv2.THRESH_BINARY)
        thresh = thresh.astype(np.uint8)
        return thresh

    def generate(self, dataset, binarize:bool=True):
        resize_to = T.Resize(self.resize_to)
        pairs = []
        for i in range(len(dataset)):
            input_img=dataset[i][0]
            target=np.squeeze(dataset[i][1].numpy().astype('int8')) if binarize else np.squeeze(dataset[i][1].numpy())
            resize_back=T.Resize((input_img.shape[1], input_img.shape[2]))
            input_img=resize_to(input_img).unsqueeze(0).to(self.device)
            gen_img=resize_back(self.model(input_img).detach().cpu().squeeze(0))
            gen_img=self.gen2binary(gen_img.squeeze().numpy(), self.threshold, self.max_value) if binarize else gen_img.squeeze().numpy()
            pairs.append((gen_img, target))
        return pairs

