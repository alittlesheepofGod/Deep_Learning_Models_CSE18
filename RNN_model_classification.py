# import libraries
import torch 

# check available device 

# device configuration 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device 

# download MNIST dataset in local system 
from torchvision import datasets 
from torchvision.transforms import ToTensor 

train_data = datasets.MNIST(
    root = 'data', 
    train = True, 
    transform = ToTensor(), 
    download =True,
)

test_data = datasets.MNIST(
    root = 'data', 
    train = False,
    transform = ToTensor()
)

# print train_data and test_data size 
print(train_data)
print(test_data)
print('train data size: ', train_data.data.size())
print('test data size : ', test_data.data.size())

# plot one train_data 
import matplotlib.pyplot as plt 

plt.imshow(train_data.data[0], cmap='gray')
plt.title('%i' % train_data.targets[0])
plt.show()

# plot multiple train_data 
figure = plt.figure(figsize=(10, 8))
cols, rows = 5, 5 
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train_data), size=(1,)).item()
    img, label = train_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(label)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

# preparing data for training with DataLoaders 

from torch.utils.data import DataLoader 

loaders = {
    'train' : torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True, num_workers=1),
    'test' : torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=True, num_workers=1)
}

loaders

# define the Recurrent Neural Network model 
from torch import nn 
import torch.nn.functional as F 

# declare hyper-parameters 
sequence_length = 28 
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.01

# create a class 

# step 1: create a class called RNN and we have to add PyTorch's base class (nn.module) for all neural network modules. 
class RNN(nn.Module):
    pass 

model = RNN().to(device)
print(model)

# step 2: add constructor in RNN class 
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
        pass

    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Passing in the input and hidden state into the model and  obtaining outputs
        out, hidden = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        #Reshaping the outputs such that it can be fit into the fully connected layer
        out = self.fc(out[:, -1, :])
        return out
        
        pass

pass

model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

print(model)

loss_func = nn.CrossEntropyLoss()
loss_func

from torch import optim
optimizer = optim.Adam(model.parameters(), lr = 0.01)   
optimizer

def train(num_epochs, model, loaders):
    print(f"num_epochs: {num_epochs}")
    print(f"model: {model}")
    print(f"loaders['train']: {loaders['train']}")
    
    print("Started epoch: ")
    for epoch in range(num_epochs):
        print("epoch: ", epoch)
        
        pass
    print("Ended epoch: ")
    
    
    pass
train(num_epochs, model, loaders)

def train(num_epochs, model, loaders):
        
    # Train the model
    total_step = len(loaders['train'])
        
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders['train']):
            
            images = images.reshape(-1, sequence_length, input_size).to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = loss_func(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
                pass
        
        pass
    pass
train(num_epochs, model, loaders)

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in loaders['test']:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total = total + labels.size(0)
        correct = correct + (predicted == labels).sum().item()
print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

sample = next(iter(loaders['test']))
imgs, lbls = sample

test_output = model((imgs[:10].view(-1, 28, 28)).to(device))
# predicted = torch.max(test_output, 1)[1].data.numpy().squeeze()
predicted = torch.max(test_output, 1)[1]
labels = lbls[:10].numpy()
print(f"Predicted number: {predicted}")
print(f"Actual number: {labels}")

