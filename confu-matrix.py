import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from torchvision.datasets import ImageFolder
from run2DCNNtorch2 import ConvNet
from config import *

from confuse_matrix import plot_confusion_matrix

model = ConvNet(nb_classes)
# model = models.resnet18(pretrained=False)
model.load_state_dict(torch.load('./models/best.pt'))

img_width, img_height = nb_residues, nb_residues
valid_data_dir = 'data-md/valid/'

nb_valid_samples = int(0.2 * nb_systems * total_prod_steps * 0.002 / stride)
batch_size = 512

valid_transform = transforms.Compose([
    transforms.Resize((img_width, img_height)),
    transforms.Grayscale(1),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.5], std=[0.5])
])

valid_dataset = ImageFolder(valid_data_dir, transform=valid_transform)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

valid_pred = []
model.eval()
with torch.no_grad():
    for inputs, _ in valid_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        valid_pred.extend(predicted.cpu().numpy())

valid_pred = np.array(valid_pred)

cf_matrix = confusion_matrix(valid_dataset.targets, valid_pred)
print(cf_matrix)
cf_matrix = cf_matrix / cf_matrix.astype(float).sum(axis=1)
print(cf_matrix)

labels = ['I', 'II', 'III', 'IV','V','VI']
save_path = './models/conf-matrix.jpg'
# 绘制混淆矩阵图
plot_confusion_matrix(cf_matrix, labels, save_path)

# plt.imshow(cf_matrix, vmin=0, vmax=1, cmap='Blues')
# cbar = plt.colorbar()
# for l in cbar.ax.yaxis.get_ticklabels():
#     l.set_weight('bold')
#     l.set_fontsize(24) 

# plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
# plt.xlabel('Predicted Class', fontsize=24, fontweight='bold')
# plt.ylabel('True Class', fontsize=24, fontweight='bold')
# plt.savefig('../pha/models/conf-matrix.jpg')