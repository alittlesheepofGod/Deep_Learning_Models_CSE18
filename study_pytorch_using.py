# from re import X
# import torch 
# import numpy as np 

# data = [[1, 2], [3, 4]]
# x_data = torch.tensor(data)

# np_array = np.array(data)
# x_np = torch.from_numpy(np_array)

# x_ones = torch.ones_like(x_data) # retain the properties of x_data 
# print(f"Ones Tensor: \n {x_ones} \n")

# x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
# print(f"Random Tensor: \n {x_rand} \n")


# import sound.effects.echo 


import torch 
import torchvision 
import torchvision.transforms as transforms 

transform = transforms.Compose(
    [transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4 

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import matplotlib.pyplot as plt 
import numpy as np 

# functions to show an image 

def imshow(img):
    img = img / 2 + 0.5   # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get some random training images 
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images 
imshow(torchvision.utils.make_grid(images))

# print labels 
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

# define a convolutional neural network 
import torch.nn as nn 
import torch.nn.functional as F 

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x 

net = Net()

# define loss function and optimizer : classification cross-entropy and SGD with momentum 
import torch.optim as optim 

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)

# train the network 
# we have to loop over our data iterator, and feed the inputs to the network and optimize 
for epoch in range(2):   # loop over the dataset multiple times 

    running_loss = 0.0 
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data 

        # zero the parameter gradients 
        optimizer.zero_grad()

        # forward + backward + optimize 
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics 
        running_loss += loss.item()
        if i%2000 == 1999:  # print every 2000 mini-batches
            print(f'[{epoch+1}, {i + 1:5d}] loss: {running_loss / 2000: .3f}')
            running_loss = 0.0 

print('Finished Training')


