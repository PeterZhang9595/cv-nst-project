## 规定ui接口的格式
### 使用gradio搭建界面
- 由于gr.Image() 组件要求PIL格式，从简化ui代码角度，希望各个模块的函数可以直接接受PIL格式的图像输入，返回PIL格式的图像输出，即Image对象。   
- 所以各个函数需要内部完成PIL格式与tensor格式的转换工作，已经一些预处理工作。
- 同时对于一些固定的预训练模型，直接在函数对应模块内直接加载，不通过ui传递模型参数。
- 如果有一些可调节的超参数，可以通过ui传递。
- 对于real_time部分，我想的是如果有多个训练好的模型,设计ui时可以通过下拉菜单选择不同的模型进行风格迁移，当然传入的也是模型名称字符串（或者编号），由函数内部完成模型加载和图像处理工作。（当然如果只有一个训练好的模型就算了）

- 关于real_time部分的说明：real_time暂时只有一个可用的训练好的模型，保存在save文件夹中，使用时输入图像和模型名称，其中模型名称不需要输入save/前缀和.pth后缀。模型的加载过程会在函数内进行，返回PIL格式的图像。由于训练集太大，代码中没有下载训练集，即仅凭给出的代码无法进行模型训练。如果有训练需求请参考real_time_style_transfer内部的README.


## 环境配置
首先从Github下载代码到本地。
1. 配置虚拟环境
   ```bash
     conda create --name cv python=3.10
     conda activate cv
   ```
2. 安装GPU版本的pytorch
   ```bash
     pip install torch>=2.4.0 torchvision>=0.19.0 torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
3. 安装其它依赖
  ```bash
    pip install -r requirements.txt
  ```
