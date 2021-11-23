import torchvision
import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import transforms, models
from torch_lr_finder import LRFinder as TLRFinder
from torchvision.utils import make_grid
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt


class ImageClassificationPT:
    def __init__(self, path, size):
        self.path = path
        self.size = size

    def load_split_train_test(self, valid_size):
        train_transforms = transforms.Compose([transforms.Resize(self.size),
                                               transforms.ToTensor(),
                                               # Augmentation block
                                               transforms.RandomVerticalFlip(0.4),
                                               transforms.RandomHorizontalFlip(0.4),
                                               transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0,
                                                                      hue=0),
                                               ])
        val_transforms = transforms.Compose([transforms.Resize(self.size),
                                             transforms.ToTensor(),
                                             # Augmentation block
                                             transforms.RandomVerticalFlip(0.4),
                                             transforms.RandomHorizontalFlip(0.4),
                                             transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0,
                                                                    hue=0),
                                             ])
        test_transforms = transforms.Compose([transforms.Resize((self.size)),
                                              transforms.ToTensor(),
                                              ])

        train_data = torchvision.datasets.ImageFolder(self.path + '/train',
                                                      transform=train_transforms)
        val_data = torchvision.datasets.ImageFolder(self.path + '/train',
                                                    transform=val_transforms)
        test_data = torchvision.datasets.ImageFolder(self.path + '/test',
                                                     transform=test_transforms)

        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))
        np.random.shuffle(indices)
        train_idx, val_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        trainloader = torch.utils.data.DataLoader(train_data,
                                                  sampler=train_sampler, batch_size=64)
        valloader = torch.utils.data.DataLoader(val_data,
                                                sampler=val_sampler, batch_size=64)
        testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

        return trainloader, valloader, testloader

    def visualize_classification(self, trainloader):
        i = 1
        for batch_idx, (inputs, labels) in enumerate(trainloader):
            grid = torchvision.utils.make_grid(inputs, nrow=5)
            plt.imshow(transforms.ToPILImage()(grid))
            plt.savefig('results/pytorch/augmented_images_part' + str(i))
            plt.show()
            i += 1

    def pretrained_model(self, model, unfreeze, trainloader, valloader, testloader, train_epochs):
        device = torch.device("cpu")

        # Architecture part
        if model == 'resnet':
            basemodel = models.resnet50(pretrained=True)
            optimizer = optim.Adam(basemodel.fc.parameters(), lr=0.003)
        elif model == 'densenet':
            basemodel = models.densenet161(pretrained=True)
            optimizer = optim.Adam(basemodel.classifier.parameters(), lr=0.003)
        else:
            raise ValueError('Model not implemented yet')

        if unfreeze == 1:
            for param in basemodel.parameters():
                param.requires_grad = True

        elif unfreeze == 0:
            for param in basemodel.parameters():
                param.requires_grad = False

        # Resnet
        if model == 'resnet':
            basemodel.fc = nn.Sequential(nn.Linear(2048, 512),
                                         nn.ReLU(),
                                         nn.Dropout(0.2),
                                         nn.Linear(512, 2),
                                         nn.LogSoftmax(dim=1))
            # optimizer = optim.Adam(basemodel.fc.parameters(), lr=0.003)
        # Densenet
        else:
            basemodel.classifier = nn.Sequential(nn.Linear(2208, 512),
                                                 nn.ReLU(),
                                                 nn.Dropout(0.1),
                                                 nn.Linear(512, 2),
                                                 nn.LogSoftmax(dim=1))

            # optimizer = optim.Adam(basemodel.classifier.parameters(), lr=0.003)
        criterion = nn.NLLLoss()
        print(basemodel.to(device))
        # LR finder
        lr_finder = TLRFinder(basemodel, optimizer, criterion)
        lr_finder.range_test(trainloader, val_loader=valloader, end_lr=1, num_iter=100, step_mode="linear")
        lr_finder.plot(log_lr=False)
        lr_finder.reset()
        learning_rate = float(input('Please put the best learning rate you see on the graph '))
        # Resnet
        if model == 'resnet':
            optimizer = optim.Adam(basemodel.fc.parameters(), lr=learning_rate)
        # Densenet
        else:
            optimizer = optim.Adam(basemodel.classifier.parameters(), lr=learning_rate)

        criterion = nn.NLLLoss()

        # Validation pass function

        # Function for the validation pass
        def validation(model, validateloader, criterion):
            val_loss = 0
            val_accuracy = 0

            for images, labels in iter(valloader):
                images, labels = images.to('cpu'), labels.to('cpu')

                output = model.forward(images)
                val_loss += criterion(output, labels).item()

                probabilities = torch.exp(output)

                equality = (labels.data == probabilities.max(dim=1)[1])
                val_accuracy += equality.type(torch.FloatTensor).mean()
            return val_loss, val_accuracy

        # Train part
        epochs = train_epochs
        steps = 0
        running_loss = 0
        print_every = 10
        train_losses, test_losses = [], []
        for epoch in range(epochs):
            for inputs, labels in trainloader:
                steps += 1
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                logps = basemodel.forward(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                if steps % print_every == 0:
                    test_loss = 0
                    accuracy = 0
                    basemodel.eval()
                    with torch.no_grad():
                        for inputs, labels in testloader:
                            inputs, labels = inputs.to(device), labels.to(device)
                            logps = basemodel.forward(inputs)
                            batch_loss = criterion(logps, labels)
                            test_loss += batch_loss.item()

                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                            basemodel.eval()
                            val_loss, val_accuracy = validation(basemodel, valloader, criterion)
                    train_losses.append(running_loss / len(trainloader))
                    test_losses.append(test_loss / len(testloader))
                    print(f"Epoch {epoch + 1}/{epochs}.. "
                          f"Train loss: {running_loss / print_every:.3f}.. "
                          f"Val loss: {val_loss / len(valloader):.3f}.. "
                          f"Val accuracy: {val_accuracy / len(valloader):.3f}.. "
                          f"Test loss: {test_loss / len(testloader):.3f}.. "
                          f"Test accuracy: {accuracy / len(testloader):.3f}")
                    running_loss = 0
                    basemodel.train()
        # torch.save(model, 'aerialmodel.pth')#Can uncomment and save the model
        plt.plot(train_losses, label='Training loss')
        plt.plot(test_losses, label='Validation loss')
        plt.legend(frameon=False)
        plt.savefig('results/pytorch/plot_loss_change.png')
        plt.show()