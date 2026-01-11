import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torchvision.models import vgg19,VGG19_Weights
from torchvision.transforms import InterpolationMode

import copy
import numpy as np
import torchvision
import cv2

# 图像预处理
def preprocess_content(image_name,device):
    # 读取文件路径
    content_imsize = (512,512) if torch.cuda.is_available() else (256,256)
    content_loader = transforms.Compose([
        transforms.Resize(content_imsize,interpolation=InterpolationMode.LANCZOS),
        transforms.ToTensor()
])
    image = Image.open(image_name)

    # 在最前面添加一个维度，变成pytorch标准的[B,C,H,W]形式
    image = content_loader(image).unsqueeze(0)
    image = image.to(device,torch.float)

    return image
def preprocess_style(image_name,device):
    style_imsize = (512,512) if torch.cuda.is_available() else (256,256)
    style_loader = transforms.Compose([
    transforms.Resize(style_imsize,interpolation=InterpolationMode.LANCZOS),
    transforms.ToTensor()
])
    image = Image.open(image_name)
    image = style_loader(image).unsqueeze(0)
    image = image.to(device,torch.float)
    return image

# 把生成的tensor形式的图片转化为PIL格式进行显示
def img_show(img_tensor,width,height,title=None):
    image = img_tensor.cpu().clone()
    image = image.squeeze(0)
    image = torch.clamp(image,0,1)
    # 把向量转化回PIL图像
    unloader = transforms.ToPILImage()
    #plt.ion()
    image = unloader(image)
    image = image.resize((width,height),resample=Image.Resampling.LANCZOS)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.show(block=True)

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        # 用 register_buffer 让 mean / std 自动随 model.to(device)
        self.register_buffer("mean", mean.clone().detach().view(1,-1,1,1))
        self.register_buffer("std",  std.clone().detach().view(1,-1,1,1))

    def forward(self, image):
        return (image - self.mean) / self.std
        
# 计算内容损失
class ContentLoss(nn.Module):
    def __init__(self,target,mask=None):
        super(ContentLoss,self).__init__()
        # target是对比图的对应层特征，我们不希望它加入计算图，所以要用detach
        self.target = target.detach()
        if mask is not None:
            self.mask = mask
        else:
            self.mask = None

    def forward(self,input):
        h,w = input.shape[2],input.shape[3]
        if self.mask is not None:
            curr_mask = F.interpolate(self.mask,size=(h,w),mode='bilinear')
            self.loss = F.mse_loss(input*curr_mask,self.target*curr_mask)
        else:
            self.loss = F.mse_loss(input,self.target)
        return input

# 计算风格损失
def gram_matrix(input):
    B,C,H,W = input.shape
    features = input.view(B*C,H*W)
    G = torch.mm(features,features.t())
    return G.div(C*B*H*W)

class StyleLoss(nn.Module):
    def __init__(self,target,mask=None):
        super(StyleLoss,self).__init__()
        self.target = gram_matrix(target).detach()
        if mask is not None:
            self.mask = mask
        else:
            self.mask = None
        

    def forward(self,input):
        h, w = input.shape[2], input.shape[3]
        if self.mask is not None:
            curr_mask = F.interpolate(self.mask, size=(h, w), mode='bilinear')
            masked_input = input * (1 - curr_mask)
            G = gram_matrix(masked_input)
            self.loss = F.mse_loss(self.target,G)
        else:
            G = gram_matrix(input)
            self.loss = F.mse_loss(self.target,G)
        return input

# 导入模型，在VGG19的特征提取层加入我们前面定义的两个loss
def get_model_with_losses(cnn,style_image,content_image,device,mask=None):
    # 我们所有提取的层都是relu
    content_layers = ["relu_4_2"]
    style_layers = ["relu_1_1","relu_2_1","relu_3_1","relu_4_1","relu_5_1"]
    content_losses = []
    style_losses = []
    rgb_mean = torch.tensor([0.485,0.456,0.406], device=device)
    rgb_std  = torch.tensor([0.229,0.224,0.225], device=device)
    # 创建一个空容器
    normalization = Normalization(rgb_mean,rgb_std)
    model = nn.Sequential(normalization)

    i = 1
    conv_index = 0
    # 请注意，在原版的VGG中，不存在BatchNormalization层
    for layer in cnn.children():
        if isinstance(layer,nn.Conv2d):
            conv_index+=1
            name = f"conv_{i}_{conv_index}"
        elif isinstance(layer,nn.ReLU):
            name = f"relu_{i}_{conv_index}"
            # 创建一个新的张量，这样有利于后面计算loss
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer,nn.MaxPool2d):
            name = f"pool_{i}"
            i+=1
            conv_index = 0
        else:
            raise RuntimeError('unrecognized layer')
        model.add_module(name,layer)


        # 请注意，我们这里存入losses列表中的并不是loss的实际值，而是一系列存储了loss实际值的loss层
        if name in content_layers:
           target = model(content_image)
           content_loss = ContentLoss(target,mask)
           model.add_module(f"content_loss{i}",content_loss)
           content_losses.append(content_loss)

        if name in style_layers:
            target = model(style_image)
            style_loss = StyleLoss(target,mask)
            model.add_module(f"style_loss{i}",style_loss)
            style_losses.append(style_loss)

    for k in range(len(model)-1,-1,-1):
        if isinstance(model[k],ContentLoss) or isinstance(model[k],StyleLoss):
            break
    
    # 这里我们为了节省计算效率，截断model到最后一个loss层
    model = model[:(k+1)]

    return model,content_losses,style_losses


class LaplacianLayer(nn.Module):
    def __init__(self,device,channels=3):
        super(LaplacianLayer,self).__init__()
        kernel = torch.tensor([
            0,-1,0,
            -1,4,-1,
            0,-1,0
        ],dtype=torch.float32,device=device)

        kernel = kernel.view(1,1,3,3) 
        # 在原论文中，作者指出，应当对RGB三个通道分别用laplacian卷积
        kernel = kernel.repeat(channels,1,1,1) # shape:[3,1,3,3]
        
        # 这里使用depthwise卷积，即对每个输入通道使用一个独立的卷积核，只对本通道卷积，
        # 不跨通道混合信息
        self.conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            groups=channels,
            bias=False,
            padding=1,
            padding_mode='reflect'
        ).to(device)

        with torch.no_grad():
            self.conv.weight.copy_(kernel)
        self.conv.weight.requires_grad = False


    def forward(self,input):
        return self.conv(input)

class LaplacianLoss(nn.Module):
    def __init__(self,target,device,channels=3):
        super(LaplacianLoss,self).__init__()
        self.channels = channels
        
        self.target = target.clone()
        self.target = self.target.sum(dim=1,keepdim=True)
        self.target = self.target.detach()

    def forward(self,input):
        temp_input = input.clone()
        temp_input = temp_input.sum(dim=1,keepdim=True)
        self.loss = F.mse_loss(temp_input,self.target)
        return input


def get_model_with_lapstyle_losses(content_image,device):
    lap_losses = []
    rgb_mean = torch.tensor([0.485,0.456,0.406], device=device)
    rgb_std  = torch.tensor([0.229,0.224,0.225], device=device)
    normalization = Normalization(rgb_mean,rgb_std)
    model = nn.Sequential(normalization)
    
    layer = nn.AvgPool2d(kernel_size=4,stride=4)
    name = "average_pooling"
    model.add_module(name,layer)

    laplacian = LaplacianLayer(device)
    name = "laplacian operator"
    model.add_module(name,laplacian)

    target = model(content_image)
    lap_loss = LaplacianLoss(target,device)
    name = "laplacian loss"
    model.add_module(name,lap_loss)

    lap_losses.append(lap_loss)

    return model,lap_losses


# 生成一张图片的掩码，使用canny边缘检测方法
def generate_edge_mask(img_path,device,target_size=(512,512)):
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray,(5,5),0)

    # 生成边缘
    edges = cv2.Canny(blurred,15,30)

    kernel = np.ones((5,5),np.uint8)

    # 这一步可以把canny检测到的细小缺口补上，让整个图像边缘更加的连贯
    closed = cv2.morphologyEx(edges,cv2.MORPH_CLOSE,kernel)

    # 填充轮廓，得到一个实心面
    contours,_ = cv2.findContours(closed,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    mask_np = np.zeros_like(gray)
    cv2.drawContours(mask_np,contours,-1,255,thickness=cv2.FILLED)

    mask_tensor = torch.from_numpy(mask_np).float() / 255.0
    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0) #shape:[1,1,H,W]
    mask_tensor = F.interpolate(mask_tensor,size=target_size,mode='nearest')

    return mask_tensor.to(device)



# 定义一个对图片进行梯度下降的优化器
def get_input_optimizer(input_image):
    optimizer = optim.LBFGS([input_image],lr = 0.1)
    return optimizer

def run_nst_lapstyle(cnn,content_image,style_image,input_image,device,mask=None,num_steps=300,style_weight=1e7,content_weight=1,laplacian_weight=1e3):
    print('Building the style transfer model...')
    if mask is not None:
        gaty_model,content_losses,style_losses = get_model_with_losses(cnn,style_image,content_image,device,mask)
    else:
        gaty_model,content_losses,style_losses = get_model_with_losses(cnn,style_image,content_image,device)
    laplacian_model,laplacian_losses = get_model_with_lapstyle_losses(content_image,device)
    input_image.requires_grad_(True)
    gaty_model.eval()
    gaty_model.requires_grad_(False)
    laplacian_model.requires_grad_(False)

    print("Begin optimizing")

    optimizer = get_input_optimizer(input_image)
    run = [0]
    while run[0] <= num_steps:
        def closure():
            with torch.no_grad():
                input_image.clamp_(0,1)

            optimizer.zero_grad()
            gaty_model(input_image)
            laplacian_model(input_image)
            style_score = 0
            content_score = 0
            laplacian_score = 0
            
            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss
            for ll in laplacian_losses:
                laplacian_score += ll.loss

            style_score *= style_weight
            content_score *= content_weight
            laplacian_score *= laplacian_weight
            loss = style_score + content_score + laplacian_score
            loss.backward()

            run[0] += 1
            if run[0] % 25 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f} Laplacian Loss: {:4f}'.format(
                    style_score.item(), content_score.item(),laplacian_score.item()))
                print()
            
            return style_score + content_score + laplacian_score
        optimizer.step(closure)

    return input_image

def run_nst_gatys(cnn,content_image,style_image,input_image,device,mask=None,num_steps=300,style_weight=1e7,content_weight=1,laplacian_weight=1e3):
    print('Building the style transfer model...')
    if mask is not None:
        gaty_model,content_losses,style_losses = get_model_with_losses(cnn,style_image,content_image,device,mask)
    else:
        gaty_model,content_losses,style_losses = get_model_with_losses(cnn,style_image,content_image,device)
    #laplacian_model,laplacian_losses = get_model_with_lapstyle_losses(content_image)
    input_image.requires_grad_(True)
    gaty_model.eval()
    gaty_model.requires_grad_(False)
    #laplacian_model.requires_grad_(False)

    print("Begin optimizing")

    optimizer = get_input_optimizer(input_image)
    run = [0]
    while run[0] <= num_steps:
        def closure():
            with torch.no_grad():
                input_image.clamp_(0,1)

            optimizer.zero_grad()
            gaty_model(input_image)
            #laplacian_model(input_image)
            style_score = 0
            content_score = 0
            #laplacian_score = 0
            
            for sl in style_losses:
                style_score += sl.loss 
            for cl in content_losses:
                content_score += cl.loss
            #for ll in laplacian_losses:
                #laplacian_score += ll.loss

            style_score *= style_weight
            content_score *= content_weight
            #laplacian_score *= laplacian_weight
            loss = style_score + content_score 
            loss.backward()

            run[0] += 1
            if run[0] % 25 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()
            
            return style_score + content_score
        optimizer.step(closure)
    return input_image


def read_and_resize(path, size=512):
    img = cv2.imread(path)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LANCZOS4)
    return img

to_tensor = transforms.ToTensor()
def cv2_to_tensor(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = to_tensor(img).unsqueeze(0).to(device,torch.float)
    return img

def pyramid_neural_transfer(model,content_image_path,style_image_path,device,style_weight=1e7):
    content_512 = read_and_resize(content_image_path, 512)
    style_512   = read_and_resize(style_image_path, 512)

    content_256 = cv2.pyrDown(content_512)
    style_256 = cv2.pyrDown(style_512)

    content_128 = cv2.pyrDown(content_256)
    style_128 = cv2.pyrDown(style_256)

    contents = {
        128: cv2_to_tensor(content_128),
        256: cv2_to_tensor(content_256),
        512: cv2_to_tensor(content_512),
    }

    styles = {
        128: cv2_to_tensor(style_128),
        256: cv2_to_tensor(style_256),
        512: cv2_to_tensor(style_512),
    }

    input_128 = contents[128].clone().requires_grad_(True)
    output_128 = run_nst_gatys(
        model,contents[128],styles[128],input_128,device,style_weight=style_weight
    )

    input_256 = F.interpolate(
        output_128,size=(256,256),mode="bilinear",align_corners=False
    ).clone()
    input_256 = input_256.detach().clone().requires_grad_(True)
    output_256 = run_nst_gatys(
        model,contents[256],styles[256],input_256,device,style_weight=style_weight
    )

    input_512 = F.interpolate(
        output_256,size=(512,512),mode="bilinear",align_corners=False
    ).clone()
    input_512 = input_512.detach().clone().requires_grad_(True)
    output_512 = run_nst_gatys(
        model,contents[512],styles[512],input_512,device,style_weight=style_weight
    )

    return output_512
def build_mask_from_pil(pil_img, out_size=(512, 512)):
    """
    pil_img: PIL.Image, RGB / L / RGBA 都可以
    return: torch.Tensor [1,1,H,W]
    """
    img_np = np.array(pil_img)

    if img_np.ndim == 3:
        if img_np.shape[2] == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        elif img_np.shape[2] == 4:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGBA2GRAY)
        else:
            raise ValueError("Unsupported channel number")
    else:
        gray = img_np  

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 15, 30)

    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    mask_np = np.zeros_like(gray)
    cv2.drawContours(mask_np, contours, -1, 255, thickness=cv2.FILLED)

    mask = torch.from_numpy(mask_np).float() / 255.0
    mask = mask.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

    mask = F.interpolate(mask, size=out_size, mode="nearest")

    return mask
# 传入的都是image格式
def run_neural_style_transfer_ui(content_image,style_image,use_laplacian=False,use_mask=False,style_weight=1e7):
    device = torch.device("cuda" if torch.cuda.is_available()else "cpu")
    content_imsize = (512,512) if torch.cuda.is_available() else (256,256)
    style_imsize = (512,512) if torch.cuda.is_available() else (256,256)
    content_loader = transforms.Compose([
        transforms.Resize(content_imsize,interpolation=InterpolationMode.LANCZOS),
        transforms.ToTensor()
    ])
    style_loader = transforms.Compose([
        transforms.Resize(style_imsize,interpolation=InterpolationMode.LANCZOS),
        transforms.ToTensor()
    ])

    width, height = content_image.size

    mask = build_mask_from_pil(content_image).to(device)

    content_image = content_loader(content_image).unsqueeze(0)
    content_image = content_image.to(device,torch.float)
    style_image = style_loader(style_image).unsqueeze(0)
    style_image = style_image.to(device,torch.float)
    input_image = content_image.clone()

    cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval().to(device)

    if use_mask and use_laplacian:
        output = run_nst_lapstyle(cnn,content_image,style_image,input_image,device,mask,style_weight=style_weight)
    elif use_mask and not use_laplacian:
        output = run_nst_gatys(cnn,content_image,style_image,input_image,device,mask,style_weight=style_weight)
    elif not use_mask and use_laplacian:
        output = run_nst_lapstyle(cnn,content_image,style_image,input_image,device,style_weight=style_weight)
    elif not use_mask and not use_laplacian:
        output = run_nst_gatys(cnn,content_image,style_image,input_image,device,style_weight=style_weight)

    image = output.cpu().clone()
    image = image.squeeze(0)
    image = torch.clamp(image,0,1)
    # 把向量转化回PIL图像
    unloader = transforms.ToPILImage()
    #plt.ion()
    image = unloader(image)
    image = image.resize((width,height),resample=Image.Resampling.LANCZOS)
    return image

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available()else "cpu")
    temp = Image.open("./images/palace.jpg")
    width, height = temp.size  
    style_image_path = "./images/starry_night.jpg"
    content_image_path = "./images/palace.jpg"
    style_image = preprocess_style(style_image_path,device)
    content_image = preprocess_content(content_image_path,device)
    cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval().to(device)

    # 定义一系列参数
    use_mask = True
    use_laplacian = True

    input_image = content_image.clone()
    mask = generate_edge_mask(content_image_path,device)
    if use_mask and use_laplacian:
        output = run_nst_lapstyle(cnn,content_image,style_image,input_image,device,mask)
    elif use_mask and not use_laplacian:
        output = run_nst_gatys(cnn,content_image,style_image,input_image,device,mask)
    elif not use_mask and use_laplacian:
        output = run_nst_lapstyle(cnn,content_image,style_image,input_image,device)
    elif not use_mask and not use_laplacian:
        output = run_nst_gatys(cnn,content_image,style_image,input_image,device)

    plt.figure()
    img_show(output, width,height,title='Output Image')

    """
    temp = Image.open("./images/hoovertowernight.jpg")
    width, height = temp.size  
    style_image_path = "./images/starry_night.jpg"
    content_image_path = "./images/hoovertowernight.jpg"
    cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
    output = pyramid_neural_transfer(cnn,content_image_path,style_image_path,1e6)
    plt.figure()
    img_show(output, width,height,title='Output Image')
    """
   
