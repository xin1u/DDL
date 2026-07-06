import math     
import numpy as np

import torch
from torchvision.utils import make_grid

from PIL import Image
                                                    
                                                                  



def compute_psnr(images, labels):

    batch, _, _, _ = images.size()
    PSNR = 0
    for i in range(batch):
        PSNR += psnr(images[i] * 255, labels[i] * 255)

    PSNR = PSNR / batch

    return PSNR


def psnr(img1, img2):

    img1 = img1.cpu().detach().numpy()
    img2 = img2.cpu().detach().numpy()
    img1 = np.transpose(np.float64(img1), (1, 2, 0))
    img2 = np.transpose(np.float64(img2), (1, 2, 0))
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)         
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])                  
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))            
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))            
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
                                                                            
    return img_np.astype(out_type)

def calculate_psnr_imgs(img1, img2):
                                       
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def single_forward(model, inp):
    with torch.no_grad():
        model_output = model(inp)
        if isinstance(model_output, list) or isinstance(model_output, tuple):
            output = model_output[0]
        else:
            output = model_output
    output = output.data.float().cpu()
    return output

def flipx4_forward(model, inp):
            
    output_f = single_forward(model, inp)

            
    output = single_forward(model, torch.flip(inp, (-1, )))
    output_f = output_f + torch.flip(output, (-1, ))
            
    output = single_forward(model, torch.flip(inp, (-2, )))
    output_f = output_f + torch.flip(output, (-2, ))
                       
    output = single_forward(model, torch.flip(inp, (-2, -1)))
    output_f = output_f + torch.flip(output, (-2, -1))

    return output_f / 4


def compute_ssim(images, labels):

    batch, _, _, _ = images.size()
    SSIM = 0
    for i in range(batch):

        SSIM += ssim(images[i] * 255, labels[i] * 255)

    SSIM = SSIM / batch
    return SSIM


def ssim(img1, img2):

    img1 = img1.cpu().detach().numpy()
    img2 = img2.cpu().detach().numpy()
    img1 = np.transpose(np.uint8(img1), (1, 2, 0))
    img2 = np.transpose(np.uint8(img2), (1, 2, 0))
    ssim_value = compare_ssim(img1, img2, multichannel=True)

    return ssim_value



def imgtoimg(img):

    img = img.data.float().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = np.uint8((img.clip(0, 1) * 255.).round())

    return img
 
                                   
 
                                    
              
                            
 
                                                      
 
                         
                 
 
 
                                           
 
         
                    
                                  
                          
         
                           
                           
                                      
                                                                         
                           
                                                       
                                                       
 
                        
                                 
                          
                                
                        
                                
                                                                  
                                           
                                  
                                                             
           
                                                           
 
 
                       
                            
                            
 
                                    
                                    
                                             
                                                   
 
                                                               
                                                      
                       
                       
                         
                                                                          
                                                                          
                                                                           
 
                                                                                      
                                                                                           
                            



