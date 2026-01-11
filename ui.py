# this file is to make a demo ui for our project's three models.
# 添加python路径
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
Adaptive_path = os.path.abspath(os.path.join(current_dir,'Adaptive_Style_Transfer'))
gatys_path = os.path.abspath(os.path.join(current_dir,'gatys_and_lapstyle'))
real_time_path = os.path.abspath(os.path.join(current_dir,'real_time_style_transfer'))
sys.path.append(Adaptive_path)
sys.path.append(gatys_path)
sys.path.append(real_time_path)


import gradio as gr
from Adaptive_Style_Transfer.ui import style_transfer_with_user_mask
from gatys_and_lapstyle.gatys import run_neural_style_transfer_ui
from real_time_style_transfer.test import stylize_ui # 处理图像在函数内部

from torchvision.models import vgg19, VGG19_Weights
import torch
cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

with gr.Blocks() as demo:
    gr.Markdown("# Neural Style Transfer with User Mask")

    with gr.Tabs():
        with gr.Tab("Use Adaptive Style Transfer Model"):
            gr.Markdown("Upload a content image, two style images, and draw a mask to specify foreground and background for style transfer.")
            with gr.Row():
                with gr.Column():
                    # in the left column, we place three image
                    content_input = gr.ImageEditor(
                        type="pil",
                        label="Content Image (Draw mask here, foreground in white, background in black)"
                    )
                    style1_input = gr.Image(type="pil", label="Style Image 1 (Foreground)")
                    style2_input = gr.Image(type="pil", label="Style Image 2 (Background)")
                with gr.Column():
                    # a slider and a sketchpad
                    run_button = gr.Button("Run Adaptive Style Transfer")
                    k_slider = gr.Slider(0, 1, value=0.5, step=0.01, label="Stylization Degree (k)")
                    # if use color matching 
                    alpha_slider = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="Color Matching Alpha, alpha=1 means full reserve the content color")
                    use_color_matching = gr.Checkbox(label="Use Color Matching")
                    output_image = gr.Image(type="pil", label="Output Image")

            run_button.click(
                fn=style_transfer_with_user_mask,
                inputs=[content_input, style1_input, style2_input, k_slider, use_color_matching, alpha_slider],
                outputs=output_image
            )
        with gr.Tab("Use Gatys_and_Lapstyle Model"):
            gr.Markdown("Upload a content image and a style image to perform neural style transfer using Gatys' method.")
            with gr.Row():
                with gr.Column():
                    content_input = gr.Image(type="pil", label="Content Image for Gatys")
                    style_input = gr.Image(type="pil", label="Style Image for Gatys")   
                    use_laplacian = gr.Checkbox(label="Use Laplacian Pyramid", value=False)
                    use_mask = gr.Checkbox(label="Use Mask", value=False)
                    style_weight = gr.Slider(1e6, 1e9, value=1e7, step=1e6, label="Style Weight")
                with gr.Column():
                    run_button_gatys = gr.Button("Run Gatys Style Transfer")
                    output_image_gatys = gr.Image(type="pil", label="Output Image for Gatys")
            run_button_gatys.click(
                fn=run_neural_style_transfer_ui,
                inputs=[content_input, style_input, use_laplacian, use_mask, style_weight],
                outputs=output_image_gatys
            )
        with gr.Tab("Use Real-Time Style Transfer Model"):
            gr.Markdown("Use real-time style transfer")
            with gr.Row():
                with gr.Column():
                    content_input = gr.Image(type="pil",label="Content Image for Real-Time Style Transfer")
                    model_name_input = gr.Textbox(label="Model Name (without .pth and save/ prefix)",value="real_time5e7")
                with gr.Column():
                    run_button_realtime = gr.Button("Run Real-Time Style Transfer")
                    output_image_realtime = gr.Image(type='pil',label="output Image for Real-Time Style Transfer")
            run_button_realtime.click(
                fn=stylize_ui,
                inputs=[content_input,model_name_input],
                outputs=output_image_realtime
            )
if __name__ == "__main__":
    demo.launch()