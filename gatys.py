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
import numpy
import torchvision

############# 选择设备 ###############
device = torch.device("cuda" if torch.cuda.is_available()else "cpu")
torch.set_default_device(device)

############# 加载图片 #################

# 这里我们使用GPU的时候，默认让图像短边大小为512，长边按照等比例进行缩放若否则使用128以节省计算量
imsize = (512,512) if torch.cuda.is_available() else (128,128)
rgb_mean = torch.tensor([0.485,0.456,0.406])
rgb_std = torch.tensor([0.229,0.224,0.225])
loader = transforms.Compose([
        transforms.Resize(imsize,interpolation=InterpolationMode.LANCZOS),
        transforms.ToTensor()
])

# 图像预处理
def preprocess(image_name):
    # 读取文件路径
    image = Image.open(image_name)

    # 在最前面添加一个维度，变成pytorch标准的[B,C,H,W]形式
    image = loader(image).unsqueeze(0)
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
    def __init__(self):
        super(Normalization,self).__init__()
        self.mean = torch.tensor(rgb_mean).view(-1,1,1)
        
        self.std = torch.tensor(rgb_std).view(-1,1,1)
    def forward(self,image):
        return (image - self.mean) / self.std
        
# 计算内容损失
class ContentLoss(nn.Module):
    def __init__(self,target):
        super(ContentLoss,self).__init__()
        # target是对比图的对应层特征，我们不希望它加入计算图，所以要用detach
        self.target = target.detach()

    def forward(self,input):
        self.loss = F.mse_loss(input,self.target)
        return input

# 计算风格损失
def gram_matrix(input):
    B,C,H,W = input.shape
    features = input.view(B*C,H*W)
    G = torch.mm(features,features.t())
    return G.div(C*B*H*W)

class StyleLoss(nn.Module):
    def __init__(self,target):
        super(StyleLoss,self).__init__()
        self.target = gram_matrix(target).detach()

    def forward(self,input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(self.target,G)
        return input

# 导入模型，在VGG19的特征提取层加入我们前面定义的两个loss
def get_model_with_losses(cnn,style_image,content_image):
    # 我们所有提取的层都是relu
    content_layers = ["relu_4_2"]
    style_layers = ["relu_1_1","relu_2_1","relu_3_1","relu_4_1","relu_5_1"]
    content_losses = []
    style_losses = []

    # 创建一个空容器
    normalization = Normalization()
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
           content_loss = ContentLoss(target)
           model.add_module(f"content_loss{i}",content_loss)
           content_losses.append(content_loss)

        if name in style_layers:
            target = model(style_image)
            style_loss = StyleLoss(target)
            model.add_module(f"style_loss{i}",style_loss)
            style_losses.append(style_loss)

    for k in range(len(model)-1,-1,-1):
        if isinstance(model[k],ContentLoss) or isinstance(model[k],StyleLoss):
            break
    
    # 这里我们为了节省计算效率，截断model到最后一个loss层
    model = model[:(k+1)]

    return model,content_losses,style_losses


class LaplacianLayer(nn.Module):
    def __init__(self,channels=3):
        super(LaplacianLayer,self).__init__()
        kernel = torch.tensor([
            0,-1,0,
            -1,4,-1,
            0,-1,0
        ],dtype=torch.float32)

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
        )

        with torch.no_grad():
            self.conv.weight.copy_(kernel)
        self.conv.weight.requires_grad = False


    def forward(self,input):
        return self.conv(input)

class LaplacianLoss(nn.Module):
    def __init__(self,target,channels=3):
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


def get_model_with_lapstyle_losses(content_image):
    lap_losses = []
    normalization = Normalization()
    model = nn.Sequential(normalization)
    
    layer = nn.AvgPool2d(kernel_size=4,stride=4)
    name = "average_pooling"
    model.add_module(name,layer)

    laplacian = LaplacianLayer()
    name = "laplacian operator"
    model.add_module(name,laplacian)

    target = model(content_image)
    lap_loss = LaplacianLoss(target)
    name = "laplacian loss"
    model.add_module(name,lap_loss)

    lap_losses.append(lap_loss)

    return model,lap_losses





# 定义一个对图片进行梯度下降的优化器
def get_input_optimizer(input_image):
    optimizer = optim.LBFGS([input_image],lr = 0.1)
    return optimizer

def run_neural_sytle_transfer(cnn,content_image,style_image,input_image,num_steps=500,style_weight=1e6,content_weight=1,laplacian_weight=100):
    print('Building the style transfer model...')
    gaty_model,content_losses,style_losses = get_model_with_losses(cnn,style_image,content_image)
    laplacian_model,laplacian_losses = get_model_with_lapstyle_losses(content_image)
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

if __name__ == "__main__":
    temp = Image.open("./images/megan.png")
    width, height = temp.size  
    style_image = preprocess("./images/flowers.png")
    content_image = preprocess("./images/megan.png")
    cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()

    input_image = content_image.clone()
    output = run_neural_sytle_transfer(cnn,content_image,style_image,input_image)

    plt.figure()
    img_show(output, width,height,title='Output Image')


   
