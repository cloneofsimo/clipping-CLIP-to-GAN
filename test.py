import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import utils as vutils

import os
import random
import argparse
from tqdm import tqdm
from random import randint

from GAN_models import Generator
import clip


# Load the model
perceptor, preprocess = clip.load('ViT-B/32')


if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)

    inputs_list = ["This is an image of young woman, she is african american. This is an image of young woman, she is african american.",
        "This is an image of old man, he is wearing glasses. This is an image of old man, he is wearing glasses.",
        "This is an image of young boy. He is 2 years old. He is just a baby. This is an image of young boy. He is 2 years old. He is just a baby.",
        "This is an image of old woman. She is from France. This is an image of old woman. She is from France.",
        "This is an image of a Mathematician from France. This person studies Algebraic Geometry. This is an image of a Mathematician from France. This person studies Algebraic Geometry.",
        "This is an image of a Mathematician from Russia. This person studies Differential Geometry. This is an image of a Mathematician from Russia. This person studies Differential Geometry."]
    N = 5

    name = str(hash(inputs_list[N])%1007)

    txt_tok = clip.tokenize(inputs_list[N])
    emb_txt = perceptor.encode_text(txt_tok.cuda())
    
    perceptor, preprocess = clip.load('ViT-B/32')
    device = torch.device('cuda:0')
    noise_dim = 256
    noise = nn.Embedding(1, noise_dim).to(device)
    noise.weight.parameters = torch.randn(1, noise_dim)
    
    EPOCHS = 10
    BATCH_SIZE = 10
    CUTS = 5
    im_size = 512
    
    Gen_Model = Generator( ngf=64, nz=noise_dim, nc=3, im_size= im_size )
    Gen_Model.to(device)
    ckpt = './models/all_100000.pth'
    checkpoint = torch.load(ckpt, map_location=lambda a,b: a)
    Gen_Model.load_state_dict(checkpoint['g'])
    Gen_Model.to(device)
    opt = optim.SGD(noise.parameters(), lr = 1e-1, weight_decay= 1e-5)
    
    for epoch in range(EPOCHS):
        
        with torch.no_grad():
            gens = Gen_Model(noise.weight)[0]
            print(noise.weight.mean(), noise.weight.std())
            vutils.save_image(gens.add(1).mul(0.5), 
                f"results/res{name + str(epoch)}.png")
        
        for _ in tqdm(range(BATCH_SIZE)):
           
            gens = Gen_Model(noise.weight)[0]
            cuts = [gens]
            # for offset in [(randint(0, 40), randint(0, 40)) for _ in range(CUTS)]:
            #     x, y = offset
            #     cuts.append(gens[:, :, x:x + 512 - 40, y:y + 512 - 40])
            #Creates regularizing effect, 
            gens = torch.cat(cuts, dim = 0)
            gens = torch.nn.functional.interpolate(gens, (224,224), mode='bilinear')
            emb_txt = perceptor.encode_text(txt_tok.cuda())

            emb_img = perceptor.encode_image(gens)
            opt.zero_grad()
            loss = -1000* torch.cosine_similarity(emb_txt, emb_img, dim = -1).mean()
            loss.backward()
            opt.step()
        
        

    ## results to gifs, which is, completely useless i know.