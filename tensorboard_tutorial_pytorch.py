<<<<<<< HEAD
# import 
=======
# imports
>>>>>>> 8f7b0f9b27576ffcb3e051348cec3f5d3f0abe85
import matplotlib.pyplot as plt 
import numpy as np 

import torch 
import torchvision 
<<<<<<< HEAD
import torchvision.transforms as transforms 
=======
import torchvision.transforms as transforms
>>>>>>> 8f7b0f9b27576ffcb3e051348cec3f5d3f0abe85

import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 

<<<<<<< HEAD
# transforms
transform = transforms.Compose(
    [transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,))])

# datasets
=======

# transforms
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))])  # chuẩn hóa dữ liệu

# datasets 
>>>>>>> 8f7b0f9b27576ffcb3e051348cec3f5d3f0abe85
trainset = torchvision.datasets.FashionMNIST('./data', download=True, train=True, transform=transform)
testset = torchvision.datasets.FashionMNIST('./data', download=True, train=False, transform=transform)

# dataloaders 
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

<<<<<<< HEAD
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

# plot image 
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5   # unnormalize 
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap='Greys')
=======
# constant for classes 
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

# helper function to show an image 
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5    # unnormalize 
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
>>>>>>> 8f7b0f9b27576ffcb3e051348cec3f5d3f0abe85
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


<<<<<<< HEAD
# we define a similar model architecture from the tutrial 
=======
# define model architecture 
>>>>>>> 8f7b0f9b27576ffcb3e051348cec3f5d3f0abe85
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
<<<<<<< HEAD
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
=======
        self.fc1 = nn.Linear(16*4*4, 120)
>>>>>>> 8f7b0f9b27576ffcb3e051348cec3f5d3f0abe85
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
<<<<<<< HEAD
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

# we define the same optimizer and criterion 
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# TensorBoard setup 
from torch.utils.tensorboard import SummaryWriter 

# default `log_dir` is "runs" - we'll be more specific here 
writer = SummaryWriter('runs/fashion_mnist_experiment_1')  # create folder

# 2. Wrtiting to TensorBoard 

# get some random training images 
dataiter = iter(trainloader)
images, labels = dataiter.next()
=======
        x = x.view(-1, 16 * 4 *4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x 

net = Net()

# we define the same optimizer and criterion from before 
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 1.tensorboard setup 

# SummaryWriter : key object for writing information to TensorBoard. 
from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here 
writer = SummaryWriter('runs/fashion_mnist_experiment_1')  # create a runs/fashion_mnist_experiment_1 folder.


# 2. writing to tensorboard 

# let's write an image to our TensorBoard - specifically, a grid - using make_grid 

# get some random training images 
detailer = iter(trainloader)
images, labels = detailer.next()
>>>>>>> 8f7b0f9b27576ffcb3e051348cec3f5d3f0abe85

# create grid of images 
img_grid = torchvision.utils.make_grid(images)

<<<<<<< HEAD
# show images
=======
# show images 
>>>>>>> 8f7b0f9b27576ffcb3e051348cec3f5d3f0abe85
matplotlib_imshow(img_grid, one_channel=True)

# write to tensorboard 
writer.add_image('four_fashion_mnist_images', img_grid)






<<<<<<< HEAD



=======
>>>>>>> 8f7b0f9b27576ffcb3e051348cec3f5d3f0abe85
