# test our trained model on some images


import torch
from torchvision import transforms
from torchvision.transforms.functional import resize
from PIL import Image
from pathlib import Path
from net import VGGEncoder, Decoder, StyleTransferNet
from torchvision.models import vgg19, VGG19_Weights
import argparse
from function import segment_foreground_background

# -------------------------
# 配置
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')


content_path = "test_content.jpg"   # 你的内容图
style_path1   = "test_style3.jpg"     # 你的风格图
style_path2  = "test_style.jpg"     # 你的第二个风格图
output_path  = "output.jpg"         # 保存结果

decoder_path = "./path/decoder.pth"  # 训练好的 decoder 权重

image_size = 512  # 最大边长

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

# -------------------------
# 图片 transform
# -------------------------
transform = transforms.Compose([
    transforms.Resize(image_size),
    # transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
])

def load_image(path):
    img = Image.open(path).convert("RGB")
    img = transform(img).unsqueeze(0)  # [1,3,H,W]
    return img.to(device)

def save_image(tensor, path):
    tensor = tensor.clamp(0, 1)   # 限制到 [0,1]
    img = transforms.ToPILImage()(tensor.squeeze(0).cpu())
    img.save(path)

argparser = argparse.ArgumentParser()
argparser.add_argument('--content', type=str, default=content_path, help='Path to the content image')
argparser.add_argument('--style1', type=str, default=style_path1, help='Path to the first style image')
argparser.add_argument('--style2', type=str, default=None, help='Path to the second style image')
argparser.add_argument('--output', type=str, default=output_path, help='Path to save the output image')
args = argparser.parse_args()


# -------------------------
# Network
# -------------------------
# VGG Encoder
vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
encoder = VGGEncoder(vgg).to(device).eval()

# Decoder
decoder = Decoder().to(device)
decoder.load_state_dict(torch.load(decoder_path, map_location=device))

# Style Transfer Network
net = StyleTransferNet(encoder, decoder).to(device).eval()

# -------------------------
# Load images
# -------------------------


# -------------------------
# Forward pass (AdaIN)
# -------------------------
with torch.no_grad():
    # now we can generate image with different degree of stylization by changing k
    # judge if we have style2 , if so , we do different style transfer on different parts
    content = load_image(args.content)
    style1 = load_image(args.style1)
    if args.style2 == None:
        generated = net.generate(content, style1, k=1)
    else:
        style2 = load_image(args.style2)
        foreground, background, mask = segment_foreground_background(args.content)
        foreground_pil = Image.fromarray(foreground)
        background_pil = Image.fromarray(background)
        fore = transform(foreground_pil).unsqueeze(0).to(device)
        back = transform(background_pil).unsqueeze(0).to(device)
        # print(fore.shape, back.shape)
        
        mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]
        # print(mask.shape)
        mask = resize(mask, size=back.shape[2:])  # 调整 mask 大小以匹配 fore/back 大小
        # print(mask.shape)

        generated = net.generate1(fore,back, style1,style2,mask,k=1)
    
    # 反归一化
    generated = generated * torch.tensor(std).view(1,3,1,1).to(device) + torch.tensor(mean).view(1,3,1,1).to(device)
    # convert to [0,1]
    generated = generated.clamp(0, 1)

# -------------------------
# Save output
# -------------------------
save_image(generated, output_path)
print(f"Stylized image saved to {output_path}")

