import torch
from torchvision import transforms
from model import TransformerNet
import argparse

def stylize_ui(content_image, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    content_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    content_image = content_transform(content_image).unsqueeze(0).to(device)
    model_path=f'real_time_style_transfer/save/{model_name}.pth'
    with torch.no_grad():
        style_model = TransformerNet()
        style_model.load_state_dict(torch.load(model_path,map_location=device))
        style_model.to(device)
        output = style_model(content_image).cpu()
        output = (output + 1.0) / 2.0 
        output = output.clamp(0, 1).cpu().squeeze(0)
    from torchvision.transforms import ToPILImage
    img = ToPILImage()(output)
    return img

def stylize(content_image, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    content_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    content_image = content_transform(content_image).unsqueeze(0).to(device)
    model_path=f'save/{model_name}.pth'
    with torch.no_grad():
        style_model = TransformerNet()
        style_model.load_state_dict(torch.load(model_path))
        style_model.to(device)
        output = style_model(content_image).cpu()
        output = (output + 1.0) / 2.0 
        output = output.clamp(0, 1).cpu().squeeze(0)
    from torchvision.transforms import ToPILImage
    img = ToPILImage()(output)
    return img

if __name__=="__main__":
    parse=argparse.ArgumentParser()
    parse.add_argument('--content',type=str,required=True,help='path to train')
    parse.add_argument('--checkpoint',type=str,required=True,help='checkpoint_save_dir')
    args=parse.parse_args()
    stylize(args.content,args.checkpoint)