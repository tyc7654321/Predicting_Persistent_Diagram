from Dataloader import *
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time
import matplotlib.pyplot as plt
from PerUnet import *
import torch
import sys


epochs = 10
lr = 0.001
batch_size = 32

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=None)
new_trainset = CIFAR10dataset(trainset)

trainloader = torch.utils.data.DataLoader(new_trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=None)
new_testset = CIFAR10dataset(testset)

testloader = torch.utils.data.DataLoader(new_testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

def init_weights(m):
    if (type(m) == nn.Conv2d) or (type(m) == nn.ConvTranspose2d) or (type(m) == nn.Linear) or (type(m) == nn.Conv3d):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.zeros_(m.bias.data)

model = PInet()

model.apply(init_weights)

criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=lr)

use_gpu = torch.cuda.is_available()
if use_gpu:
    model = model.cuda()


def progressBar(i, max, text):
    bar_size = 30
    j = i / max
    sys.stdout.write('\r')
    sys.stdout.write(
        f"[{'=' * int(bar_size * j):{bar_size}s}] {int(100 * j)}%  {text}")
    sys.stdout.flush()

def train():

    train_loss_list = list()
    val_loss_list = list()
    test_loss_list = list()

    for epoch in range(epochs):
        print("epoch:", epoch)
        ts = time.time()

        train_loss = 0
        val_loss = 0
        test_loss = 0

        for iter, (X, Y, labels) in enumerate(trainloader):
            
            # training on the first 40000 images
            while iter <= 0.2 * len(new_trainset):
                optimizer.zero_grad()

                inputs_image = torch.as_tensor(X, device=torch.device('cuda'))
                GT_image = torch.as_tensor(Y, dtype=torch.float32, device=torch.device('cuda'))

                outputs = model(inputs_image)
                loss = criterion(outputs, GT_image)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
            
            progressBar(iter + 1, 0.2 * len(new_trainset) / batch_size,
                            "Train Progress")

            # validating on the rest 10000 images
            while iter > 0.2 * len(new_trainset) and iter <= 0.4 * len(new_trainset):
                inputs_image = torch.as_tensor(X, device=torch.device('cuda'))
                GT_image = torch.as_tensor(Y, dtype=torch.float32, device=torch.device('cuda'))

                outputs = model(inputs_image)
                loss = criterion(outputs, GT_image)

                val_loss += loss.item()

                progressBar(iter + 1, 0.2 * len(new_trainset) / batch_size,
                        "Validation Progress")

            if iter > 0.4 * len(new_trainset):
                break

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        # testing on 10000 images
        for iter, (X, Y, labels) in enumerate(testloader):

            inputs_image = torch.as_tensor(X, device=torch.device('cuda'))
            GT_image = torch.as_tensor(Y, dtype=torch.float32, device=torch.device('cuda'))

            outputs = model(inputs_image)
            loss = criterion(outputs, GT_image)

            test_loss += loss.item()

            progressBar(iter + 1, len(new_testset) / batch_size,
                        "Test Progress")

        test_loss_list.append(test_loss)

        plot_performance(train_loss_list, val_loss_list, test_loss_list)

        model.train()

        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))

    return

def plot_performance(train_loss_list, val_loss_list, test_loss_list):
    plt.plot(train_loss_list, label="Train Loss")
    plt.plot(val_loss_list, label="Validation Loss")
    plt.plot(test_loss_list, label="Test Loss")
    plt.show()


if __name__ == "__main__":
    train()