import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms,datasets
import torchvision.models as models
from PIL import Image
from model import VGGFeatures,TransformerNet
import argparse
import torch.utils.data as data
from torch.utils.data import DataLoader

def stylize(content_image_path, model_path, output_image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    content_image = Image.open(content_image_path)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    content_image = content_transform(content_image).unsqueeze(0).to(device)

    with torch.no_grad():
        style_model = TransformerNet()
        style_model.load_state_dict(torch.load(model_path))
        style_model.to(device)
        output = style_model(content_image).cpu()
        output = (output + 1.0) / 2.0 
        output = output.clamp(0, 1).cpu().squeeze(0)
    from torchvision.transforms import ToPILImage
    img = ToPILImage()(output)
    img.save(f'output/{output_image_path}.png')
    return img

if __name__=="__main__":
    parse=argparse.ArgumentParser()
    parse.add_argument('--content_path',type=str,required=True,help='path to train')
    parse.add_argument('--checkpoint',type=str,required=True,help='checkpoint_save_dir')
    parse.add_argument('--save_name',default='res',)
    args=parse.parse_args()
    stylize(args.content_path,f'save/{args.checkpoint}',args.save_name)