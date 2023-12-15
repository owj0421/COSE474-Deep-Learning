import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
# from vgg16_full import *
from resnet50_skeleton import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

# Image Preprocessing
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# CIFAR-10 Dataset
train_dataset = torchvision.datasets.CIFAR10(root='../osproj/data/',
                                             train=True,
                                             transform=transform_train,
                                             download=False) # Change Download-flag "True" at the first excution.

test_dataset = torchvision.datasets.CIFAR10(root='../osproj/data/',
                                            train=False,
                                            transform=transform_test)


# data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=100,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=100,
                                          shuffle=False)
###########################################################
# Choose model
model = ResNet50_layer4().to(device)
PATH = './resnet50_epoch285.ckpt' # test acc would be almost 80

# model = vgg16().to(device)
# PATH = './vgg16_epoch250.ckpt'  # test acc would be almost 85
##############################################################
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint)

# Train Model
# Hyper-parameters
num_epochs = 1  # students should train 1 epoch because they will use cpu
learning_rate = 0.001

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# For updating learning rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Train the model
total_step = len(train_loader)
current_lr = learning_rate

for epoch in range(num_epochs):

    model.train()
    train_loss = 0

    for batch_index, (images, labels) in enumerate(train_loader):
        # print(images.shape)
        images = images.to(device)  # "images" = "inputs"
        labels = labels.to(device)  # "labels" = "targets"

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if (batch_index + 1) % 100 == 0:
            print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                  .format(epoch + 1, num_epochs, batch_index + 1, total_step, train_loss / (batch_index + 1)))

    # Decay learning rate
    if (epoch + 1) % 20 == 0:
        current_lr /= 3
        update_lr(optimizer, current_lr)
        torch.save(model.state_dict(), './resnet50_epoch' + str(epoch+1)+'.ckpt')

# Save the model checkpoint
torch.save(model.state_dict(), './resnet50_final.ckpt')

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))