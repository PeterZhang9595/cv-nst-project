# in this net.py
# I want to implement a neural network module
# to do Arbitrary Style Transfer in Real-time
# with Adaptive Instance Normalization


import torch
import torch.nn as nn
from function import adaptive_instance_normalization as adain
from function import calc_mean_std

# we will use a vgg encoder , a adain layer 
# a decoder to reconstruct the image 
# and a same vgg for loss calculation




class AdaIN(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    # the adain layer 
    # remove the mean and std of content picture
    # and apply the mean and std of style picture
    def forward(self, content_feat, style_feat):
        c_mean, c_std = self.calc_mean_std(content_feat)
        s_mean, s_std = self.calc_mean_std(style_feat)

        normalized = (content_feat - c_mean) / c_std
        return normalized * s_std + s_mean
    # the function to calculate mean and std
    def calc_mean_std(self, feat):
        # for the feat 
        # its shape is (N, C, H, W)
        # [:2] is N,C
        # for the std and mean
        # we should calculate along H and W 
        N, C = feat.size()[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + self.eps

        # notice that at last we should view it back to (N, C, H, W)
        # just the H and W is 1 now.
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std
    
# -------------------------
# VGG Encoder (multi-layer output)
# -------------------------
class VGGEncoder(nn.Module):
    """
    Extract features from multiple VGG layers:
    relu1_1, relu2_1, relu3_1, relu4_1
    """

    # in the paper they say 
    # up to relu4_1 of a pre-trained VGG-19
    def __init__(self, vgg):
        super().__init__()
        self.layers = nn.ModuleList([
            vgg[:2],    # relu1_1 1
            vgg[2:7],   # relu2_1 6
            vgg[7:12],  # relu3_1 11
            vgg[12:21], # relu4_1 20
        ])
        # fix the encoder
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        # for different layers 
        # we will store the features of dirrerent layers
        feats = []
        for layer in self.layers:
            x = layer(x)
            feats.append(x)
        return feats

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        # all in all, return back to the intial image size 
        # from the reversed vgg path

        self.model = nn.Sequential(
            # mostly mirrors the encoder 
            # replace all pooling layers with nearest up-sampling 
            
            # input: relu4_1 512 * H/8 * W/8

            nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=1, padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1, padding_mode='reflect'),
            nn.ReLU(inplace=True),  
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1, padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1, padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, stride=1, padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1, padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1, padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1, padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1, stride=1, padding_mode='reflect'),
        )
    def forward(self, x):
        return self.model(x)

# -------------------------
# Style Transfer Net
# -------------------------
class StyleTransferNet(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder.eval()   # VGG is fixed
        self.decoder = decoder
        self.adain = AdaIN()
        self.mse = nn.MSELoss()

    def forward(self, content, style):
        content_feats = self.encoder(content)
        style_feats = self.encoder(style)

        # AdaIN on relu4_1
        t = self.adain(content_feats[-1], style_feats[-1])
        generated = self.decoder(t)
        gen_feats = self.encoder(generated)

        # Content loss (relu4_1)
        content_loss = self.mse(gen_feats[-1], t)

        # Style loss (multiple layers) relu1_1, relu2_1, relu3_1, relu4_1
        style_loss = 0.0
        for gf, sf in zip(gen_feats, style_feats):
            style_loss += self.mse(
                self.calc_mean_std(gf)[0], self.calc_mean_std(sf)[0]
            )
            style_loss += self.mse(
                self.calc_mean_std(gf)[1], self.calc_mean_std(sf)[1]
            )

        return content_loss, style_loss
    
    def generate(self, content, style):
        content_feats = self.encoder(content)
        style_feats = self.encoder(style)

        t = self.adain(content_feats[-1], style_feats[-1])
        generated = self.decoder(t)
        return generated


    def calc_mean_std(self, feat):
        N, C = feat.size()[:2]
        feat = feat.view(N, C, -1)
        mean = feat.mean(dim=2)
        std = feat.var(dim=2, unbiased=False).sqrt()
        return mean, std



decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

# vgg19 structure 2 + 2 + 4 + 4 + 4 conv layers
# 1 + 1 + 1 linear layers (fc layers) are removed
# 16 + 3 = 19 layers in total
vgg = nn.Sequential(
    nn.Conv2d(3,3,kernel_size=1, stride=1, padding=0), # input normalization layer

    nn.Conv2d(3,64,kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
    nn.ReLU(), # relu1_1
    nn.Conv2d(64,64,kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
    nn.ReLU(), # relu1_2
    nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True),

    nn.Conv2d(64,128,kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
    nn.ReLU(), # relu2_1
    nn.Conv2d(128,128,kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
    nn.ReLU(), # relu2_2
    nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True),

    nn.Conv2d(128,256,kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
    nn.ReLU(), # relu3_1
    nn.Conv2d(256,256,kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
    nn.ReLU(), # relu3_2
    nn.Conv2d(256,256,kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
    nn.ReLU(), # relu3_3
    nn.Conv2d(256,256,kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
    nn.ReLU(), # relu3_4
    nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True),

    nn.Conv2d(256,512,kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
    nn.ReLU(), # relu4_1
    nn.Conv2d(512,512,kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
    nn.ReLU(), # relu4_2
    nn.Conv2d(512,512,kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
    nn.ReLU(), # relu4_3
    nn.Conv2d(512,512,kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
    nn.ReLU(), # relu4_4
    nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True),

    nn.Conv2d(512,512,kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
    nn.ReLU(), # relu5_1
    nn.Conv2d(512,512,kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
    nn.ReLU(), # relu5_2
    nn.Conv2d(512,512,kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
    nn.ReLU(), # relu5_3
    nn.Conv2d(512,512,kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
    nn.ReLU(), # relu5_4
    nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)

)


# vgg = nn.Sequential(
#     nn.Conv2d(3, 3, (1, 1)),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(3, 64, (3, 3)),
#     nn.ReLU(),  # relu1-1
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(64, 64, (3, 3)),
#     nn.ReLU(),  # relu1-2
#     nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(64, 128, (3, 3)),
#     nn.ReLU(),  # relu2-1
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(128, 128, (3, 3)),
#     nn.ReLU(),  # relu2-2
#     nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(128, 256, (3, 3)),
#     nn.ReLU(),  # relu3-1
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 256, (3, 3)),
#     nn.ReLU(),  # relu3-2
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 256, (3, 3)),
#     nn.ReLU(),  # relu3-3
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 256, (3, 3)),
#     nn.ReLU(),  # relu3-4
#     nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 512, (3, 3)),
#     nn.ReLU(),  # relu4-1, this is the last layer used
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(512, 512, (3, 3)),
#     nn.ReLU(),  # relu4-2
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(512, 512, (3, 3)),
#     nn.ReLU(),  # relu4-3
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(512, 512, (3, 3)),
#     nn.ReLU(),  # relu4-4
#     nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(512, 512, (3, 3)),
#     nn.ReLU(),  # relu5-1
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(512, 512, (3, 3)),
#     nn.ReLU(),  # relu5-2
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(512, 512, (3, 3)),
#     nn.ReLU(),  # relu5-3
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(512, 512, (3, 3)),
#     nn.ReLU()  # relu5-4
# )


class Net(nn.Module):
    def __init__(self, encoder, decoder):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.decoder = decoder
        self.mse_loss = nn.MSELoss()

        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # extract relu4_1 from input image
    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def forward(self, content, style, alpha=1.0):
        assert 0 <= alpha <= 1
        style_feats = self.encode_with_intermediate(style)
        content_feat = self.encode(content)
        t = adain(content_feat, style_feats[-1])
        t = alpha * t + (1 - alpha) * content_feat

        g_t = self.decoder(t)
        g_t_feats = self.encode_with_intermediate(g_t)

        loss_c = self.calc_content_loss(g_t_feats[-1], t)
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        for i in range(1, 4):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])
        return loss_c, loss_s
    def generate(self, content, style, alpha=1.0):
        assert 0 <= alpha <= 1
        style_feats = self.encode_with_intermediate(style)
        content_feat = self.encode(content)
        t = adain(content_feat, style_feats[-1])
        t = alpha * t + (1 - alpha) * content_feat

        g_t = self.decoder(t)
        return g_t