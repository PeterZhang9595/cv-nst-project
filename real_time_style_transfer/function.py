import torch
#计算gram matrix的函数
def gram_matrix(features):
    b, c, h, w = features.size()
    # 将特征平展为 (batch, channels, height * width)
    features = features.view(b, c, h * w)
    # 计算 Gram 矩阵: (b, c, hw) @ (b, hw, c) -> (b, c, c)
    # 使用 transpose(1, 2) 交换最后两个维度
    G = torch.bmm(features, features.transpose(1, 2))
    # 归一化
    return G / (c * h * w)

#定义损失函数

def content_loss(gen_feat, content_feat):
    #内容损失，定义为特征图之间的均方误差
    return torch.mean((gen_feat - content_feat) ** 2)

def style_loss(gen_feats, style_grams):
    #风格损失，定义为特征图gram矩阵间的均方误差
    loss = 0
    for  l in range(len(gen_feats)):
        G = gram_matrix(gen_feats[l])
        A = style_grams[l]
        loss += torch.mean((G - A) ** 2)
    return loss

def tv_loss(img):
    #平滑项
    w_variance = torch.sum(torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2))
    h_variance = torch.sum(torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2))
    return h_variance + w_variance

def normalize_batch(batch):
    # VGG 预训练模型需要的均值和标准差
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return (batch - mean) / std