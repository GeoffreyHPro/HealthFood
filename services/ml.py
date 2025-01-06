import os
import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

import PIL.Image as Image

classes = [
    "bananne",
    "pomme",
]

def main():
    train_dataset_path = './fruits/training/training'
    test_dataset_path = './fruits/validation/validation'

    mean = [0.4363, 0.4328, 0.3291]
    std = [0.2129, 0.2075, 0.2038]

    train_transforms = transforms.Compose([
        transforms.Resize((224,224)), 
        transforms.RandomHorizontalFlip(), 
        transforms.RandomRotation(10), 
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(std)),
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((224,224)), 
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(std)),
    ])

    train_dataset = torchvision.datasets.ImageFolder(root = train_dataset_path,transform = train_transforms)
    test_dataset = torchvision.datasets.ImageFolder(root = test_dataset_path,transform = test_transforms)
    

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 32, shuffle=True)


    resnet18_model = models.resnet18(pretrained=False)
    num_ftrs = resnet18_model.fc.in_features
    number_of_classes = 10
    resnet18_model.fc = nn.Linear(num_ftrs, number_of_classes)
    device = set_device()
    resnet18_model = resnet18_model.to(device)
    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.SGD(resnet18_model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.003)
    train_nn(resnet18_model, train_loader, test_loader, loss_fn, optimizer, 5)

    checkpoint = torch.load('model_best_checkpoint_fruits.pth.tar')

    resnet18_model = models.resnet18()
    num_ftrs = resnet18_model.fc.in_features
    number_of_classes = 10
    resnet18_model.fc = nn.Linear(num_ftrs, number_of_classes)
    resnet18_model.load_state_dict(checkpoint['model'])

    torch.save(resnet18_model, 'best_model_fruits.pth')

def train_nn(model, train_loader, test_loader, criterion, optimizer, n_epochs):
    device = set_device()
    best_acc = 0

    for epoch in range(n_epochs):
        print("Epoch number % (epoch + 1)")
        model.train()
        running_loss = 0.0
        running_correct = 0.0
        total = 0

        for data in train_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)
            
            optimizer.zero_grad()
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            running_correct += (labels==predicted).sum().item()

        epoch_loss = running_loss/len(train_loader)
        epoch_acc = 100.00 * running_correct / total

        print('Training dataset. Got %d out of %d images correctly (%.3f%%). Epoch loss: %.3f' % (running_correct, total, epoch_acc, epoch_loss))

        test_dataset_acc = evaluate_model_on_test_set(model, test_loader)

        if(test_dataset_acc > best_acc):
            best_acc = test_dataset_acc
            save_checkpoint(model, epoch, optimizer, best_acc)


    print("Finished")
    return model

def save_checkpoint(model, epoch, optimizer, best_acc):
    state = {
        "epoch": epoch + 1,
        "model" : model.state_dict(),
        "best accuracy" : best_acc,
        "optimizer" : optimizer.state_dict(),
        "comments" : "very cool model!"
    }

    torch.save(state, "model_best_checkpoint_fruits.pth.tar")


def evaluate_model_on_test_set(model, test_loader):
    model.eval()
    predicted_correctly_on_epoch = 0
    total = 0
    device = set_device()

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)

            predicted_correctly_on_epoch += (predicted == labels).sum().item()

    epoch_acc = 100.00 * predicted_correctly_on_epoch / total
    print("testing dataset. Got %d out of %d images correctly (%.3f%%)" % (predicted_correctly_on_epoch, total, epoch_acc))
    return epoch_acc

def set_device():
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    return torch.device(dev)    

def show_transformed_images(dataset):
    loader = torch.utils.data.DataLoader(dataset = dataset, batch_size=6, shuffle=True)
    batch = next(iter(loader))
    images, labels = batch

    grid = torchvision.utils.make_grid(images, nrow = 3)
    plt.figure(figsize=(11,11))
    plt.imshow(np.transpose(grid, (1,2,0)))
    print('labels:', labels)
    plt.show()



def test_image(imagePath):
    model = torch.load("best_model_fruits.pth")

    mean = [0.4363, 0.4328, 0.3291]
    std = [0.2129, 0.2075, 0.2038]

    image_transforms = transforms.Compose([
        transforms.Resize((224,224)), 
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(std)),
    ])

    return classify(model, image_transforms, imagePath, classes)


def classify(model, image_transforms, image_path, classes):
    model = model.eval()
    image = Image.open(image_path)
    image = image_transforms(image).float()
    image = image.unsqueeze(0)

    output = model(image)
    _, predicted = torch.max(output.data,1)

    print(classes[predicted.item()])
    return classes[predicted.item()]
