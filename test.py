# test our trained model on some images


import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
from net import VGGEncoder, Decoder, StyleTransferNet
from torchvision.models import vgg19, VGG19_Weights

# -------------------------
# 配置
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')


content_path = "test_content.jpg"   # 你的内容图
style_path   = "test_style.jpg"     # 你的风格图
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
content = load_image(content_path)
style   = load_image(style_path)

# -------------------------
# Forward pass (AdaIN)
# -------------------------
with torch.no_grad():
    # now we can generate image with different degree of stylization by changing k
    generated = net.generate(content, style,k=0)
    
    # 反归一化
    generated = generated * torch.tensor(std).view(1,3,1,1).to(device) + torch.tensor(mean).view(1,3,1,1).to(device)
    # convert to [0,1]
    generated = generated.clamp(0, 1)

# -------------------------
# Save output
# -------------------------
save_image(generated, output_path)
print(f"Stylized image saved to {output_path}")

