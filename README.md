# 环境配置
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

# 运行ui界面
我们精心设计了一个ui界面以进行更方便的效果复现。
在仓库主文件夹内运行```python ui.py```，即可进入ui界面。您可以从电脑本地传入图片，使用我们的模型进行神经风格迁移。
- 您可以随意传入内容图和风格图，调节一系列参数，获得Gatys-style和Lapstyle的结果。建议您下载CUDA以利用GPU获得更快的运行速度和更好的生成效果。
- 您可以传入内容图，获得real-time transfer的结果。
- 您可以同时传入内容图和风格图，获得AdaIN的结果。
