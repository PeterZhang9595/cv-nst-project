import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms,datasets
import torchvision.models as models
from PIL import Image
from function import content_loss,style_loss,tv_loss,gram_matrix,normalize_batch
from model import VGGFeatures,TransformerNet
import tqdm
import os
import argparse
import torch.utils.data as data
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

imsize = 512 if torch.cuda.is_available() else 256

transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])


def train(style_img,save_model_name,dataset_path="contentset/coco",epochs=2,lr=1e-4,content_weight=1e2,style_weight=1e8,tw_weight=1e-3):
    dataset_path='contentset/coco'
    save_model_path=f'save/{save_model_name}.pth'
    style_img=transform(style_img).unsqueeze(0).to(device)
    train_dataset = datasets.ImageFolder(dataset_path, transform)
    train_loader = DataLoader(train_dataset, batch_size=4)
    vgg=VGGFeatures().to(device)
    transformer=TransformerNet().to(device)
    optimizer=optim.Adam(transformer.parameters(),lr=lr)
    style_feature=vgg(style_img)
    gram= [gram_matrix(ft) for ft in style_feature]
    for e in tqdm.tqdm(range(epochs)):
        transformer.train()
        for batch_id, (x, _) in enumerate(train_loader):
            optimizer.zero_grad()
            x = x.to(device)
            y = (transformer(x) + 1) / 2

            y = normalize_batch(y)
            x = normalize_batch(x)
            #print(f"Output range: min={y.min().item():.4f}, max={y.max().item():.4f}")
            features_y = vgg(y)
            features_x = vgg(x)
            
            # 内容损失 (relu3_3)
            c_loss = content_loss(
            features_y[2],
            features_x[2]
            )
            s_loss = style_loss(features_y, gram)
            t_loss=tv_loss(y)
            total_loss=content_weight*c_loss+style_weight*s_loss+tw_weight*t_loss
            total_loss.backward()
            optimizer.step()
            if (batch_id + 1) % 100 == 0:
                print(f"Epoch {e}: Batch {batch_id+1}, Loss: {total_loss.item()/100}")
    torch.save(transformer.state_dict(), save_model_path)
if __name__=="__main__":
    parse=argparse.ArgumentParser()
    parse.add_argument('--content_path',type=str,default="contentset/coco",help='path to train')
    parse.add_argument('--style_img',type=str,required=True,help='style img PIL')
    parse.add_argument('--checkpoint_name',type=str,default='path',help='checkpoint_name')
    args=parse.parse_args()

    train(args.style_img,args.checkpoint_name,args.content_path,)
    
    