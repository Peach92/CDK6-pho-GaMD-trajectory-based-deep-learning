#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Suraj Srinivas <suraj.srinivas@idiap.ch>
#

""" Misc helper functions """

import cv2
import numpy as np
import subprocess

import torch
import torchvision.transforms as transforms
from config import *
from heatmap import *                                                                                   

class NormalizeInverse(transforms.Normalize):
    # Undo normalization on images

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super(NormalizeInverse, self).__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super(NormalizeInverse, self).__call__(tensor.clone())


def create_folder(folder_name):
    try:
        subprocess.call(['mkdir','-p',folder_name])
    except OSError:
        None

def save_saliency_map(image, saliency_map, filename, saliency_img, vertical_lines=None, horizontal_lines=None, v_labels=None, h_labels=None, offset=0):
    """ 
    Save saliency map on image.
    
    Args:
        image: Tensor of size (3,H,W)
        saliency_map: Tensor of size (1,H,W) 
        filename: string with complete path and file extension

    """

    image = image.data.cpu().numpy()
    saliency_map = saliency_map.data.cpu().numpy()

    #print("-------")
    #print(saliency_map)
    saliency_map = saliency_map - saliency_map.min()
    saliency_map = saliency_map / saliency_map.max()
    saliency_map = saliency_map.clip(0,1)

    width, height = nb_residues, nb_residues

    print(f"saliency map shape: {saliency_map.shape}")
    saliency_map = saliency_map.reshape(width, height)
    print(f"saliency map shape: {saliency_map.shape}")
    if saliency_map.shape[0] != nb_residues or saliency_map.shape[1] != nb_residues:
        print("error: saliency map shape is not (1, nb_residues, nb_residues)")

    contacts = np.argwhere((saliency_map >= thres))
    contacts = contacts + [1, 1]
    print(filename)
    print(*contacts)
    print("-----------\n")
    contacts = contacts + [10, 10]
    print(*contacts)

    # saliency_map = saliency_map.reshape(1, width, height)
    # saliency_map = np.uint8(saliency_map * 255).transpose(1, 2, 0)
    # saliency_map = cv2.resize(saliency_map, (224,224))

    # saliency_map = np.uint8(saliency_map * 255)
    # 调用封装函数绘制相关性矩阵图
    plot_heatmap(saliency_map, output_file=saliency_img, dpi=600, vertical_lines=vertical_lines, horizontal_lines=horizontal_lines, v_labels=v_labels, h_labels=h_labels, offset=offset)

    '''

    image = np.uint8(image * 255).transpose(1,2,0)
    # image = cv2.resize(image, (224, 224))

    # Apply JET colormap
    color_heatmap = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)
    
    # Combine image with heatmap
    img_with_heatmap = np.float32(color_heatmap) #+ np.float32(image)
    img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap)

    cv2.imwrite(filename, np.uint8(255 * img_with_heatmap))
    # saliency_img = './results/saliency.jpg'
    cv2.imwrite(saliency_img, np.uint8(saliency_map))

    '''