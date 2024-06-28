# Pix2Pix_Modern
## An updated and easy to use repository for training the Pix2Pix architecture in Pytorch

Pix2Pix is a neural architecture published in CVPR 2017 by Phillip Isola et al, https://phillipi.github.io/pix2pix/. This architecture was the state-of-the-art for multi-modal image-to-image translation tasks for many years until the more recent introduction of denoising diffusion models. Despite this, these models are still fun to explore for various problems and serve as an important benchmark in comparative studies. This is my updated pytorch implementation that I hope may be useful to others. 

This code repository pulls it's UNet backbone from recently published diffusion research. The UNet uses modern Resblocks and allows for the inclusion of spatial self-attention (compatible with Xformers). It also makes it easier to test and share configuraitons by adopting yaml configuration files paired with initialization and trainer functions. See the code in the Demo folder for a tutorial. 

To install, run: 
```
pip install .
pip install -r requirements.txt
```
You should install pytorch on your own. Also, to get the most from self-attention, install the Xformer library https://github.com/facebookresearch/xformers via:
```
(cuda 11.8 version)
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu118
(cuda 12.1 version)
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu121
```

Thats all. Feel free to fork or update this repository with pushed changes. 
