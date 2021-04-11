# **Asymmetric Upsample EDSR**

## **Motivation.**
Most recent super-resolution (SR) convnets all focus on SR with *symmetric* scaling factors, where both dimensions of an image are scaled equally. There are cases where an *asymmetric* scaling
would be advantageous. Specifically, in 3D medical imaging a single axis is sometimes under-sampled relative to the others (typically to reduce radiation exposure). This leads to irregularly shaped voxels and poor resolution along one of the three spatial dimensions, which can inhibit medical image analysis tasks undertaken by humans or computers. 

In this project I adapt an SR convnet to work with asymmetric scaling factors. I select EDSR for this task, but an asymmetric upsampling module like the one I implement here can be added to **any** SR convnet that uses so-called learned "post-upsampling" layer(s) in the tail of the network. 
___
## **Using the repo.**
I recommend reading the documentation of the [original repo](https://github.com/twtygqyy/pytorch-edsr). The basic functionaliy is preserved here.

Check out all the options in [option.py](/src/option). You can set defaults to reduce the amount of arguments you have to pass at run time. I changed a bunch of the default options so that the default EDSR model is not really EDSR but closer to EDSR "baseline" as they refer to it in the paper (which, if you read it you would have found that EDSR "baseline" is just SRResNet without batchnorm layers)

You will definitely want to change these options:
```
# Data specifications
parser.add_argument('--dir_data', type=str, default='C:\\.data\\SRDATA',
                    help='dataset directory')
parser.add_argument('--dir_demo', type=str, default='..\\test',
                    help='demo image directory')
```
Depending on your use case, you may need to create some unique dataset files containing a class similar to `\src\data\div2k.py`.
___
## **Examples.**
### **Training** from command line:

 ```
 python main.py --model EDSR --scale 2 --patch_size 96 --save DIV2K_x2 --reset --data_train 'DIV2K' --data_test 'DIV2K' --data_range 1-800/801-850 --asymmetric False
 ```
### **Testing** from command line:

```
python main.py --data_test Demo --dir_demo <folder_full_of_test_images> --pre_train <path_to_pre-trained_net> --scale 2 -- asymmetric False --test_only --save_results
```
___
## **Changes from EDSR.**

>`TLDR;` "... the modifications preserve the original EDSR with symmetric upscaling. The asymmetric modules and associated functions are switched on and off as needed with an `--asymmetric` boolean argument."

I add an **optional** `AsymmetricUpsampler()` to the [official EDSR repo for PyTorch](https://github.com/twtygqyy/pytorch-edsr). The `AsymmetricUpsampler()` module is in `/src/model/common.py` and is a replacement for the default `Upsampler()` module, to be used in the "tail" of the network when it is constructed by `/src/model/edsr.py`. 

I remove many models, benchmarks, option arguments, etc from the original repository. I removed all the GAN and perceptual loss modules. The modifications preserve the original functioning of the EDSR with symmetric upscaling. The asymmetric modules and associated functions are switched on and off as needed with an `--asymmetric` boolean argument. Symmetric upsampling will still use the original `nn.PixelShuffle()` method and associated functions.

Vanilla EDSR (and many other post-upsample style convnets) use the ["sub-pixel"](https://arxiv.org/abs/1609.05158) method of upsampling, but the PyTorch implementation (`nn.PixelShuffle()`) is not compatible with asymmetric scaling factors. This functionality could be added easily, but it is buried in the C++ code that makes up PyTorch's `Functionals`, and I'm more interested in a drop-in solution based purely on Python code at this time. 

My solution is to just have `AsymmetricUpsampler()` use a `nn.ConvTranspose2d()` layer to do the upsampling instead of the `nn.PixelShuffle()`. Since the `nn.ConvTranspose2d()` layer takes asymmetric stride, kernel, and padding arguments, it implicitly allows for asymmetric upsampling learned in-network. I also made some other modifications to the surrounding convolutional layers, but the head and ResBlock body of the EDSR convnet remain unchanged. Notably, the popular `nn.PixelShuffle()`-based upsampling is mostly computational boon. SR by properly-implemented transposed convolution and sub-pixel convolution are [functionally the same](https://arxiv.org/abs/1609.07009), so we should not expect a difference in raw SR performance.

```
class AsymmetricUpsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bias = True):
        
        m = []
        m.append(conv(n_feats, 2*n_feats, 3))
        m.append(nn.ConvTranspose2d(2*n_feats, 1, [2,3], stride = [2, 1], padding = [0, 1]))
        
        super(AsymmetricUpsampler, self).__init__(*m)
```

There are many other changes made throughout the remaining code that allow the `AsymmetricUpsampler()` to work. These are the main additions or alterations that I can remember making (beside the code purge descibed above):

0. Added `AsymmetricUpsampler()` module to `/src/model/common.py`.
1. Added `--asymmetric` (default `False`) flag to `/src/option.py`.
2. Added asymmetric patch extraction to `get_patch()` function in `/src/data/common.py`.
3. Added `asymmetric` flags and conditions to `/src/data/srdata.py`.
4. Fixed `augment()` in `/src/data/common.py` to work with asymmetric (HR) patches.
5. Changed `calc_psnr()` in `/src/utility.py` to consider entire image, no truncation.
6. Fixed a bug on Windows in the `checkpoint()` class involving file system manipulations.
7. Fixed a bug on Windows by disabling some of the multithread features.
8. Fixed a bug on Windows using the `--test_only` mode. Now it actually works.

Plus some little stuff I'm forgetting that I fixed when something broke along the way.
___