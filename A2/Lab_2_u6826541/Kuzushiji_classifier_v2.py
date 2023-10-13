import numpy as np
import tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

#! all parameters
device           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_imgs_file  = np.load("data/kmnist-train-imgs.npz")['arr_0']
train_label_file = np.load("data/kmnist-train-labels.npz")['arr_0']
valid_imgs_file  = np.load("data/kmnist-val-imgs.npz")['arr_0']
valid_label_file = np.load("data/kmnist-val-labels.npz")['arr_0']
test_imgs_file   = np.load("data/kmnist-test-imgs.npz")['arr_0']
test_label_file  = np.load("data/kmnist-test-labels.npz")['arr_0']
batch_size       = 64
epochs           = 50
lr               = 0.001
patience         = 0 # for early stopping 

#! transform 1. normalise->random flip->padding->random crop
transform = transforms.Compose([
                                transforms.Normalize(0.5, 0.5),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.Pad(4, fill=0, padding_mode='constant'), 
                                transforms.RandomCrop(28)
                                ])# transforms.RandomResizedCrop((28,28))

#! dataset and dataloader and classes
class kmnistImageDataset(Dataset):
    def __init__(self, annotations, imgs, transform=None, target_transform=None):
        self.img_labels       = torch.from_numpy(annotations).long()
        self.imgs             = torch.from_numpy(imgs).float()
        self.imgs             = self.imgs.unsqueeze(1)
        self.transform        = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image   = self.imgs[idx]
        label   = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

training_data   = kmnistImageDataset(train_label_file, train_imgs_file, transform)
validation_data = kmnistImageDataset(valid_label_file, valid_imgs_file, transform)
test_data       = kmnistImageDataset(test_label_file,  test_imgs_file,  transform)

train_dataloader = DataLoader(training_data,   batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
valid_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
test_dataloader  = DataLoader(test_data,       batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)

classes = ('お','き','す','つ','な','は','ま','や','れ','を')

#! model
class kmnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1   = nn.Conv2d(1,  32, kernel_size=5, stride=1, padding=2)
        self.conv2   = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1     = nn.Linear(64 * 7 * 7, 1024)
        self.fc2     = nn.Linear(1024, 10)
        self.pool    = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p = 0.1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
    
model = kmnistCNN()
model.to(device)

#! Define a Loss function and optimizer and lr scheduler
LossFunction = nn.CrossEntropyLoss()
optimizer    = optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=lr)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95) # using Exponential LR scheduler

#! Train the network
best_model_loss = 9999999 # current best model loss
best_accuracy   = 0       # current best model accuracy
PATH            = ""      # current best model path
# loop over the dataset multiple times
for epoch in tqdm.tqdm(range(epochs), colour='GREEN', desc='epoch loop'):  
    running_loss = 0.0 # running_loss = 100 batch loss / 100
    T_total      = 0
    T_correct    = 0
    train_loss   = 0   # train_loss = all loss / len(all data)
    for i, data in enumerate(train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss    = LossFunction(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        train_loss   += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        T_total      += labels.size(0)
        T_correct    += (predicted == labels).sum().item()
        if i % 100 == 99:    # print every 100 mini-batches
            print(f'\n[{epoch + 1}, {i + 1}] loss: {running_loss / 100}, accuracy: {100 * T_correct // T_total}%')
            running_loss = 0.0
    
    # write to tensorboard        
    writer.add_scalar('training loss',     train_loss / len(train_dataloader), epoch)
    writer.add_scalar('training accuracy', 100 * T_correct // T_total,         epoch)
    
    #validation
    running_loss    = 0.0
    validation_loss = 0.0
    total           = 0
    correct         = 0
    print(f"\nStarting validation for epoch {epoch + 1}")
    with torch.no_grad():
        # Validation loop
        for i, data in enumerate(valid_dataloader, 0):
            inputs, labels  = data[0].to(device), data[1].to(device)
            outputs         = model(inputs)
            loss            = LossFunction(outputs, labels)
            validation_loss += loss.item()
            _, predicted    = torch.max(outputs.data, 1)
            total           += labels.size(0)
            correct         += (predicted == labels).sum().item()
            
        validation_loss /= len(valid_dataloader)
        print(f"\nValidation loss and accuracy for epoch {epoch + 1} - [loss: {validation_loss}, accuracy: {100 * correct // total}%] | Current best loss and accuracy[loss: {best_model_loss}, accuracy: {best_accuracy}%]")
    
    # write to tensorboard     
    writer.add_scalar('validation loss',     validation_loss,        epoch)
    writer.add_scalar('validation accuracy', 100 * correct // total, epoch)
    writer.add_scalars('training/validation loss',     {'training_loss' : train_loss / len(train_dataloader), 'validation_loss' : validation_loss},        epoch)
    writer.add_scalars('training/validation accuracy', {'training_acc' : 100 * T_correct // T_total,         'validation_acc' : 100 * correct // total}, epoch)
    
    # early stop algorithm to prevent overfitting 
    # reference: https://www.kaggle.com/general/178486
    if validation_loss < best_model_loss:
        best_model_loss = validation_loss
        best_accuracy   = 100 * correct // total
        patience        = 0
        # Save best model 
        PATH = f"checkpoints/cifar_net_best.pth"
        torch.save(model.state_dict(), PATH)
    else:
        # if patience > 9, training will stop
        patience += 1
        if patience > 9:
            print(f"\nEarly stopping for epoch {epoch + 1} - [best_model_loss: {best_model_loss}, best_accuracy: {best_accuracy}%]")
            break
        
    lr_scheduler.step()
        
#! Test the network on the test data
BestModel = kmnistCNN()
BestModel.load_state_dict(torch.load(PATH))
print(f'best_accuracy: {best_accuracy}, best_model_loss: {best_model_loss}')

#! Test the model on the test set
correct = 0
total   = 0
# since we're not training, we don't need to calculate the gradients for out outputs
with torch.no_grad():
    for data in test_dataloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = BestModel(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total   += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"\nAccuracy of the network on the test images: {100 * correct // total}%")
writer.close()