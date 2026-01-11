
## Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization 

### Goal
- speed the style transfer in real-time
- use a novel dadptive instance normalization(AdaIn)
layer that aligns the mean and variance of the content features with those of the style features.
- allows flexible user controls such as content-style trade-off, style interpolation, clolor & spatial controls, all using a single feed-forward neural network! 

### core implement
- a VGG to get a situation of content and style
- a AdaIn to remove the style of the content picture
and add the style of the style picture.
- a decoder to return the situation to a picture
- a VGG the same as the first part to get the situation, and use for the loss compute with the adain layer's output(content and style).

### some points may be done to control while running
- content-style trade-off already complete!
- style interpolation I don't think it help a lot.
- spatial control trying ... basically complete , now I think is better.
- color control already done!


### some improvement --- more advanced network architectures
- residual architecture
- skip connections
- more feature statistics such as correlation alignment or histogram matching


### how to use it
```bash
python test.py --content test_content.jpg --style1 test_style.jpg
python ui.py 
```
```
This is a simple example, and the output image will be saved as output.jpg.
ui.py is a simple ui to help you do the content-style-tradeoff,spatial and color control.
```
> --content: path to content image
>
> --style1: path to style image
>
> --output: path to save the output image
>
> --style2: path to second style image for style interpolation, but here we only use 
> default segmentation to do the spatial control, if you want to use your own segmentation masks, please run python ui.py and paint them by yourself.