import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert (feat.size()[0] == 3)
    assert (isinstance(feat, torch.FloatTensor))
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std


def _mat_sqrt(x):
    U, D, V = torch.svd(x)
    return torch.mm(torch.mm(U, D.pow(0.5).diag()), V.t())


def coral(source, target):
    # assume both source and target are 3D array (C, H, W)
    # Note: flatten -> f

    source_f, source_f_mean, source_f_std = _calc_feat_flatten_mean_std(source)
    source_f_norm = (source_f - source_f_mean.expand_as(
        source_f)) / source_f_std.expand_as(source_f)
    source_f_cov_eye = \
        torch.mm(source_f_norm, source_f_norm.t()) + torch.eye(3)

    target_f, target_f_mean, target_f_std = _calc_feat_flatten_mean_std(target)
    target_f_norm = (target_f - target_f_mean.expand_as(
        target_f)) / target_f_std.expand_as(target_f)
    target_f_cov_eye = \
        torch.mm(target_f_norm, target_f_norm.t()) + torch.eye(3)

    source_f_norm_transfer = torch.mm(
        _mat_sqrt(target_f_cov_eye),
        torch.mm(torch.inverse(_mat_sqrt(source_f_cov_eye)),
                 source_f_norm)
    )

    source_f_transfer = source_f_norm_transfer * \
                        target_f_std.expand_as(source_f_norm) + \
                        target_f_mean.expand_as(source_f_norm)

    return source_f_transfer.view(source.size())

# try to implement the image divide and different style transfer on different parts


def segment_foreground_background(image_path):

    # Convert PIL image to OpenCV format
    if isinstance(image_path, str):
        img = cv2.imread(image_path)
    else:
        img = np.array(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


    mask = np.zeros(img.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # 假设前景在中间
    h, w = img.shape[:2]
    rect = (40,20, w-1, h-1)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 15, cv2.GC_INIT_WITH_RECT)

    result_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    result_mask = cv2.GaussianBlur(result_mask, (15, 15), 0)
    foreground = img * result_mask[:, :, np.newaxis]
    background = img * (1 - result_mask[:, :, np.newaxis])
    return foreground, background,result_mask


# based rgb color space
def color_matching(content,style):
    content_np = np.array(content).astype(np.float32)
    style_np = np.array(style).astype(np.float32)

    content_mean = np.mean(content_np, axis=(0, 1), keepdims=True)
    content_std = np.std(content_np, axis=(0, 1), keepdims=True)

    style_mean = np.mean(style_np, axis=(0, 1), keepdims=True)
    style_std = np.std(style_np, axis=(0, 1), keepdims=True)
    # 将style的颜色预处理为content的颜色分布
    matched_style = (style_np - style_mean) / style_std * content_std + content_mean
    matched_style = np.clip(matched_style, 0, 255).astype(np.uint8)
    return Image.fromarray(matched_style)


def color_matching_local(content,style,grid_size=8,alpha=0.5,eps=1e-5):
    content_np = np.array(content).astype(np.float32)
    style_np = np.array(style).astype(np.float32)

    h, w, c = content_np.shape
    grid_h,grid_w = h // grid_size, w // grid_size

    matched_style = np.zeros_like(style_np)

    for i in range(grid_size):
        for j in range(grid_size):
            content_patch = content_np[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w, :]
            style_patch = style_np[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w, :]

            content_mean = np.mean(content_patch, axis=(0, 1), keepdims=True)
            content_std = np.std(content_patch, axis=(0, 1), keepdims=True) + eps

            style_mean = np.mean(style_patch, axis=(0, 1), keepdims=True)
            style_std = np.std(style_patch, axis=(0, 1), keepdims=True) + eps

            matched_patch = (style_patch - style_mean) / style_std * content_std + content_mean
            matched_style[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w, :] = matched_patch*alpha + style_patch*(1-alpha)


    matched_style = np.clip(matched_style, 0, 255).astype(np.uint8)
    return Image.fromarray(matched_style)   


def histogram_matching_opencv(source, target,alpha=0.5):
    """使用 OpenCV 进行直方图匹配"""
    result = np.zeros_like(source)
    
    for i in range(3):  # 对每个通道
        # 计算直方图
        hist_source = cv2.calcHist([source], [i], None, [256], [0, 256])
        hist_target = cv2.calcHist([target], [i], None, [256], [0, 256])
        
        # 计算累积直方图
        cdf_source = hist_source.cumsum()
        cdf_target = hist_target.cumsum()
        
        # 归一化
        cdf_source = (cdf_source - cdf_source.min()) * 255 / (cdf_source.max() - cdf_source.min())
        cdf_target = (cdf_target - cdf_target.min()) * 255 / (cdf_target.max() - cdf_target.min())
        
        # 创建映射表
        lut = np.interp(cdf_source, cdf_target, range(256))
        lut = np.clip(lut, 0, 255).astype(np.uint8)
        
        # 应用查找表
        result[:,:,i] = cv2.LUT(source[:,:,i], lut)

        # add a alpha blending between source and result
        result[:,:,i] = cv2.addWeighted(source[:,:,i], 1-alpha, result[:,:,i], alpha, 0)
    
    return result