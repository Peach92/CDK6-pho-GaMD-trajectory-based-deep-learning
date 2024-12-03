#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Suraj Srinivas <suraj.srinivas@idiap.ch>
#

""" Compute saliency maps of images from dataset folder
    and dump them in a results folder """

import torch
from torchvision import datasets, transforms, utils, models
import os

# Import saliency methods
from saliency.fullgrad import FullGrad
from saliency.simple_fullgrad import SimpleFullGrad
from saliency.smooth_fullgrad import SmoothFullGrad

from saliency.gradcam import GradCAM
from saliency.grad import InputGradient
from saliency.smoothgrad import SmoothGrad

from misc_functions import *

from run2DCNNtorch2 import ConvNet
from config import *

# PATH variables
PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
# dataset = PATH + 'dataset/'
dataset = './test_img/train/'


batch_size = 1

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")


data_transforms = transforms.Compose([
    transforms.Grayscale(1),
    transforms.ToTensor(),
    #transforms.Normalize((0.5,), (0.5,))
])

# load valid data and transform
valid_dataset = datasets.ImageFolder(root=dataset, transform=data_transforms)

# get sub_dir name and the label mapping
class_to_idx = valid_dataset.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}

# Dataset loader for sample images
sample_loader = torch.utils.data.DataLoader(valid_dataset, batch_size= batch_size, shuffle=False)


# unnormalize = NormalizeInverse(mean = [0.485, 0.456, 0.406],
#                            std = [0.229, 0.224, 0.225])

# Use pretrained ResNet-18 provided by PyTorch
# model = models.resnet18(pretrained=True)
model = ConvNet(nb_classes)
model.load_state_dict(torch.load('./models/best.pt'))
model = model.to(device)

# Initialize saliency methods
saliency_methods = {
# FullGrad-based methods
# 'fullgrad': FullGrad(model),
# 'simple_fullgrad': SimpleFullGrad(model),
# 'smooth_fullgrad': SmoothFullGrad(model),

# Other saliency methods from literature
# 'gradcam': GradCAM(model),
#'inputgrad': InputGradient(model),
'smoothgrad': SmoothGrad(model)
}

def compute_saliency_and_save():
    #vertical_lines = []
    #horizontal_lines = []
    #vertical_lines = [13, 23, 40, 64, 79, 92, 104, 133, 146,  173, 203, 226, 253]
    #horizontal_lines = [13, 23, 40, 64, 79, 92, 104, 133, 146, 173, 203, 226, 253]

    vertical_lines = [0,41, 61, 94, 107, 141, 153, 180, 216, 260, 281]
    v_labels = [r'$\beta$1-3', r'$\alpha$C', r'$\beta$4-5',  r'$\alpha$D',r'$\alpha$E',' ', 'T-loop',r'$\alpha$F',r'$\alpha$G ',r'$\alpha$H',' ']

    horizontal_lines = [0,41, 61, 94, 107, 141, 153, 180, 216, 260,281]
    h_labels = [r'$\beta$1-3', r'$\alpha$C', r'$\beta$4-5', r'$\alpha$D' ,r'$\alpha$E',' ', 'T-loop',r'$\alpha$F ',r'$\alpha$G ',r'$\alpha$H',' ']
    offset = 10
    for batch_idx, (data, _) in enumerate(sample_loader):
        data = data.to(device).requires_grad_()
        
        labels = [idx_to_class[idx.item()] for idx in _]

        # Compute saliency maps for the input data
        for s in saliency_methods:
            saliency_map = saliency_methods[s].saliency(data)

            # Save saliency maps
            for i in range(data.size(0)):
                filename = save_path + labels[0] + str( (batch_idx+1) * (i+1))
                # image = unnormalize(data[i].cpu())
                image = data[i].cpu()
                save_saliency_map(image, saliency_map[i], filename + '_' + s + '.jpg', filename+'_'+s+'_gray'+'.jpg', vertical_lines=vertical_lines, horizontal_lines=horizontal_lines,  v_labels=v_labels, h_labels=h_labels , offset=offset)


if __name__ == "__main__":
    # Create folder to saliency maps
    save_path = PATH + 'results/'
    create_folder(save_path)
    compute_saliency_and_save()
    print('Saliency maps saved.')







