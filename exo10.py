# import libraries
import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from model import LeNet, Net
import sys
from torchsummary import summary


# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 20

# convert data to torch.FloatTensor
transform = transforms.ToTensor()

# choose the training and test datasets
train_data = datasets.MNIST(root='data', train=True,download=True, transform=transform)
                                   
test_data = datasets.MNIST(root='data', train=False,download=True, transform=transform)
                                
train_data, val_data = torch.utils.data.random_split(train_data,[50000,10000])
                   

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, num_workers=num_workers)

print(len(train_data))
print(len(val_data))
print(len(train_loader))
print(len(test_loader))
print(len(val_loader))
import matplotlib.pyplot as plt
    
# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy()

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
  ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
  ax.imshow(np.squeeze(images[idx]), cmap='gray')
  # print out the correct label for each image .item() gets the value contained in a Tensor
  ax.set_title(str(labels[idx].item()))    
plt.show()

img = np.squeeze(images[1])
print(img.shape)
print(np.max(img))
print(np.min(img))



model = Net()
print(model)
summary(model,(1,28,28),20) #input is size of an image (28x28) and batch size is 20
optimization= False
if optimization == False:
  sys.exit(0);
## Specify loss and optimization functions

# specify loss function
criterion = nn.CrossEntropyLoss()

# specify optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


# number of epochs to train the model
n_epochs = 15  # suggest training between 20-50 epochs


for epoch in range(n_epochs):
    # monitor training loss
    train_loss = 0.0
    val_loss   = 0.0    
    ###################
    # train the model #
    ###################
    model.train() # prep model for training

    for data, target in train_loader:
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        #la fonction de cout est moyennee sur la taille du batch. On multiplie ici par la taille du batch. 
        #c est juste pour l affichage
        train_loss += loss.item()*data.size(0)


    # print training statistics 
    # calculate average loss over an epoch
    train_loss = train_loss/len(train_loader.dataset)
    #len(train_loader.dataset) correspond au nombre total d echantillons donc on a une moyenne par echantillon.


    model.eval() # prep model for *evaluation*
    for data, target in val_loader:
      # forward pass: compute predicted outputs by passing inputs to the model
      output = model(data)
      # calculate the loss
      loss = criterion(output, target)
      # update test loss 
      val_loss += loss.item()*data.size(0)
    
    # calculate and print avg test loss
    val_loss = val_loss/len(val_loader.dataset)
    
    print('Epoch: {} \tTraining Loss: {:.6f} \tTest Loss: {:.6f}\n'.format(
        epoch+1, 
        train_loss,
        val_loss
        ))
        
torch.save(model,"model.pt")        










