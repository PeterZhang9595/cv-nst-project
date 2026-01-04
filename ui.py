# in this file ,we use gradio to build a simple ui for style transfer
import torch
import gradio as gr
from PIL import Image
from net import VGGEncoder, Decoder, StyleTransferNet
from torchvision.models import vgg19, VGG19_Weights
from torchvision import transforms

# Load the pre-trained models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
decoder_path = "./path/decoder.pth"  # 训练好的 decoder 权重
vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
encoder = VGGEncoder(vgg).to(device).eval()
decoder = Decoder().to(device)
decoder.load_state_dict(torch.load(decoder_path, map_location=device))
net = StyleTransferNet(encoder, decoder).to(device).eval()
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
def transform_image(img):
    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    img = transform(img).unsqueeze(0).to(device)
    return img

def de_transform_image(tensor):
    tensor = tensor * torch.tensor(std).view(1,3,1,1).to(device) + torch.tensor(mean).view(1,3,1,1).to(device)
    tensor = tensor.clamp(0, 1)
    img = transforms.ToPILImage()(tensor.squeeze(0).cpu())
    return img

def style_transfer(content_img, style_img, k):
    content = transform_image(content_img)
    style = transform_image(style_img)
    with torch.no_grad():
        generated = net.generate(content, style, k=k)
    output_img = de_transform_image(generated)
    return output_img

# Build Gradio interface
iface = gr.Interface(   
    fn=style_transfer,
    inputs=[
        gr.Image(type="pil", label="Content Image"),
        gr.Image(type="pil", label="Style Image"),
        gr.Slider(0, 1, value=0.5, step=0.01, label="Stylization Degree (k)")
    ],
    outputs=gr.Image(type="pil", label="Output Image"),
    title="Neural Style Transfer",
    description="Upload a content image and a style image to perform neural style transfer."
)

if __name__ == "__main__":
    iface.launch()