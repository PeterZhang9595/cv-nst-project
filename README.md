# cv-nst-project
你好，这里是本次大作业的Gatys-style及Lapstyle实现的分支。

## 关于前端接口
必须设定两个全局变量：
1. ```cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()```作为参考模型。
2. ```device = torch.device("cuda" if torch.cuda.is_available()else "cpu")```获取运行设备。并且随后设定```torch.set_default_device(device)```。
### 正常模型
在前端设计时需要两个核心变量,```use_lapstyle```作为布尔变量决定要使用的损失函数，```use_mask```作为布尔变量决定是否要使用前景背景分割。

这两个是基础的接口。
```run_neural_sytle_transfer_gatys(cnn,content_image,style_image,input_image,mask=None,num_steps=300,style_weight=1e7,content_weight=1,laplacian_weight=1e3,device=device)```

```run_neural_sytle_transfer_lapstyle(cnn,content_image,style_image,input_image,mask=None,num_steps=300,style_weight=1e7,content_weight=1,laplacian_weight=1e3,device=device)```

其中```cnn```和```device```作为全局变量传入。
由于这里面的image都是tensor格式，因此在调用API之前应该先读取图片路径，大体格式如下：
```python
    style_image_path = "./images/picasso.jpg"
    content_image_path = "./images/hoovertowernight.jpg"
    style_image = preprocess_style(style_image_path)
    content_image = preprocess_content(content_image_path)
    input_image = content_image.clone()#这里的克隆很重要
```
当选定需要mask的时候，应该调用```generate_edge_mask(img_path,target_size=(512,512),device=device)```生成一个mask的tensor传入到函数中;若否，mask默认为None，不用管。

最后返回的output也是一个张量，因此如果需要转为图片，应该调用```img_show(img_tensor,width,height,title=None)```，当然我这里的展示用的是plt，因此你可以简单改写一下。

至于几个weight，可以简单设定一个对```style weight```的滑动条，范围在$10^4~10^8$之间，默认$10^7$。其余的weight就别设置为可变模式了,保持初始值就好。

### 金字塔模型
这个金字塔模型与前面的模型是独立的。因此麻烦你把这个模型单独开辟出一个板块。
```pyramid_neural_transfer(model,content_image_path,style_image_path,device=device)```
对这个函数我进行了比较好的包装，因此你只需要处理好model和device的同时，输入内容和风格图的路径就可以了。这里的```style_weight```仍需你设定为动态调整的格式，默认为1e7。