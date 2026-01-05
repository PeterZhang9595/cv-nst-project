# in this file ,we use gradio to build a simple ui for style transfer
import torch
import gradio as gr
from PIL import Image
from net import VGGEncoder, Decoder, StyleTransferNet
from torchvision.models import vgg19, VGG19_Weights
from torchvision import transforms
from function import segment_foreground_background

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
    is_gray = img.mode == "L"
    transform_list = [transforms.Resize(512), transforms.ToTensor()]
    if not is_gray:
        transform_list.append(transforms.Normalize(mean=mean, std=std))
    transform = transforms.Compose(transform_list)
    img = transform(img).unsqueeze(0).to(device)
    return img

def de_transform_image(tensor):
    tensor = tensor * torch.tensor(std).view(1,3,1,1).to(device) + torch.tensor(mean).view(1,3,1,1).to(device)
    tensor = tensor.clamp(0, 1)
    # print(tensor.shape)
    img = transforms.ToPILImage()(tensor.squeeze(0).cpu())
    return img

def style_transfer(content_img, style_img, k):
    content = transform_image(content_img)
    style = transform_image(style_img)
    with torch.no_grad():
        generated = net.generate(content, style, k=k)
    output_img = de_transform_image(generated)
    return output_img
# try to make user select the foreground and background 
# if not user select , do fixed grabcut way
def style_transfer_with_user_mask(content,style1,style2=None,user_mask=None,k=1):
    content_pil = content
    content = transform_image(content)
    style1 = transform_image(style1)

    if style2 is None:
        with torch.no_grad():
            generated = net.generate(content,style1,k=k)
        output_img = de_transform_image(generated)
    else:
        if isinstance(user_mask, dict):
            user_mask = user_mask['composite']
        if user_mask is None:
            foreground, background, mask = segment_foreground_background(content_pil)
            foreground = transform_image(Image.fromarray(foreground))
            background = transform_image(Image.fromarray(background))
            style2 = transform_image(style2)
            with torch.no_grad():
                generated_foreground = net.generate(foreground, style1, k)
                generated_background = net.generate(background, style2, k)
                mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float().to(device)  # [1,1,H,W]
                mask = torch.nn.functional.interpolate(mask, size=generated_foreground.shape[2:], mode='bilinear', align_corners=False)
                generated = generated_foreground * mask + generated_background * (1 - mask)
            output_img = de_transform_image(generated)
        else:
            mask = transform_image(user_mask.convert("L")).to(device)  # [1,1,H,W]
            # print(mask.shape)
            # print(content.shape)
            # print(style1.shape)

            style2 = transform_image(style2)
            with torch.no_grad():
                generated_foreground = net.generate(content, style1, k)
                generated_background = net.generate(content, style2, k)
                # print(generated_foreground.shape)
                # print(generated_background.shape)
                mask = torch.nn.functional.interpolate(mask, size=generated_foreground.shape[2:], mode='bilinear', align_corners=False)
                generated = generated_foreground * mask + generated_background * (1 - mask)
                #print(generated.shape)
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

iface_with_mask = gr.Interface(
    fn = style_transfer_with_user_mask,
    inputs = [
        gr.Image(type="pil", label="Content Image"),
        gr.Image(type="pil", label="Style Image 1 (Foreground)"),
        gr.Image(type="pil", label="Style Image 2 (Background)"),
        gr.Sketchpad(type="pil",label="User Mask (Draw white for foreground, black for background)"),
        gr.Slider(0, 1, value=0.5, step=0.01, label="Stylization Degree (k)")
    ],
    outputs=gr.Image(type="pil", label="Output Image"),
    title="Neural Style Transfer with User Mask",
    description="Upload a content image, two style images, and draw a mask to specify foreground and background for style transfer."
)

with gr.Blocks() as demo:
    gr.Markdown("# Neural Style Transfer with User Mask")
    gr.Markdown("Upload a content image, two style images, and draw a mask to specify foreground and background for style transfer.")
    with gr.Row():
        with gr.Column():
            # in the left column, we place three image
            content_input = gr.Image(type="pil", label="Content Image")
            style1_input = gr.Image(type="pil", label="Style Image 1 (Foreground)")
            style2_input = gr.Image(type="pil", label="Style Image 2 (Background)")
        with gr.Column():
            # a slider and a sketchpad
            run_button = gr.Button("Run Style Transfer")
            k_slider = gr.Slider(0, 1, value=0.5, step=0.01, label="Stylization Degree (k)")
            mask_input = gr.Sketchpad(type="pil", label="User Mask (Draw white for foreground, black for background)")
            output_image = gr.Image(type="pil", label="Output Image")

    run_button.click(
        fn=style_transfer_with_user_mask,
        inputs=[content_input, style1_input, style2_input, mask_input, k_slider],
        outputs=output_image
    )


if __name__ == "__main__":
    # iface.launch()
    demo.launch()