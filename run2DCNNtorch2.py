import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
import pickle
from config import *


train_data_dir = './data-md/train/'
valid_data_dir = './data-md/valid/'
img_width, img_height = nb_residues, nb_residues
model_best = './models/best.pt'
model_last = './models/last.pt'
model_metrics = './metrics/' + 'metrics.pkl'

class ConvNet(nn.Module):
    def __init__(self, nb_classes):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * (img_width // 16) * (img_height // 16), 512)
        #self.fc1 = nn.Linear(64 * (img_width // 4) * (img_height // 4), 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.output = nn.Linear(128, nb_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.output(x)
        return x

def conv_net(train_data_dir, valid_data_dir, img_width, img_height,
            nb_classes, nb_epochs, batch_size, model_metrics):

    
    device = torch.device("cuda:7")
    #device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    
    train_transform = transforms.Compose([
        # transforms.RandomResizedCrop((img_width, img_height)),
        # transforms.RandomHorizontalFlip(),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    train_dataset = ImageFolder(root=train_data_dir, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    valid_transform = transforms.Compose([
        # transforms.Resize((img_width, img_height)),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    valid_dataset = ImageFolder(root=valid_data_dir, transform=valid_transform)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    
    model = ConvNet(nb_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    
    # define best model and best model loss
    best_loss = float('inf')
    best_model_wts =None

    # save metrics
    metrics = {'loss':[], 'val_loss':[], 'accuracy':[], 'val_accuracy':[]}

    best_loss = float('inf')
    best_model_wts = None
    
    dir = './models'
    if not os.path.exists(dir):
        os.makedirs(dir)
        print(f'folder {dir} create successfully.')
    else:
        print(f'folder {dir} exist.')
    
    dir = './metrics'
    if not os.path.exists(dir):
        os.makedirs(dir)
        print(f'folder {dir} create successfully.')
    else:
        print(f'folder {dir} exist.')

    for epoch in range(nb_epochs):
        train_loss = 0.0
        valid_loss = 0.0
        train_correct = 0
        valid_correct = 0
        
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == labels).sum().item()
        
        model.eval()
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                valid_correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader.dataset)
        valid_loss /= len(valid_loader.dataset)
        train_accuracy = train_correct / len(train_loader.dataset)
        valid_accuracy = valid_correct / len(valid_loader.dataset)

        # save metrics
        metrics['loss'].append(train_loss)
        metrics['val_loss'].append(valid_loss)
        metrics['accuracy'].append(train_accuracy)
        metrics['val_accuracy'].append(valid_accuracy)

        # save the best model
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_model_wts = model.state_dict()

        print(f"Epoch {epoch+1}/{nb_epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Valid Accuracy: {valid_accuracy:.4f}")

        # check if it's the last epoch and save the model
        if epoch == nb_epochs - 1:
            torch.save(model.state_dict(), model_last)

    # save the best model
    torch.save(best_model_wts, model_best)

    print("Training completed!")

    metrics_file = open(model_metrics, 'wb')
    pickle.dump(metrics, metrics_file)
    metrics_file.close()

if __name__ == '__main__':
    conv_net(train_data_dir, valid_data_dir, img_width, img_height,
            nb_classes, nb_epochs, batch_size, model_metrics)
