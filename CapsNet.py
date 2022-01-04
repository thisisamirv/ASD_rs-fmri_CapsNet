#!/usr/local/bin/python3

# Import packages
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from einops.layers.torch import EinMix as Mix, Rearrange
from einops import rearrange, reduce, asnumpy


# Set device
device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA version: {}".format(torch.version.cuda))
print("Number of CUDA devices: {}".format(torch.cuda.device_count()))
print("Current device is CUDA: {}".format(torch.cuda.is_available()))
print("Current device #num: {}".format(torch.cuda.current_device()))
print("Current device spec: {}".format(torch.cuda.device(0)))
print("Current device name: {}".format(torch.cuda.get_device_name(0)))


# Set global variables
epoch_num = 10
routing_iter = 3
batch_size = 128


# Squash function
def squash(x, dim):
    norm_sq = torch.sum(x**2, dim, keepdim=True)
    norm = torch.sqrt(norm_sq)
    return (norm_sq / (1.0 + norm_sq)) * (x / norm)


# PrimaryCaps + DigitCaps layers
class CapsuleLayerWithRouting(nn.Module):
    def __init__(self, in_caps, in_hid, out_caps, out_hid):
        super().__init__()
        self.input_caps2U = Mix(
            'b in_caps in_hid -> b in_caps out_caps out_hid',
            weight_shape='in_caps in_hid out_caps out_hid',
            in_hid=in_hid, in_caps=in_caps, out_hid=out_hid, out_caps=out_caps,
        )
    def forward(self, input_capsules, routing_iterations):
        U = self.input_caps2U(input_capsules)
        batch, in_caps, out_caps, out_hid = U.shape
        B = torch.zeros([batch, in_caps, out_caps], device=U.device)
        for _ in range(routing_iterations):
            # b=batch, i=input cap, o=output cap, h=hidden dim of output cap
            C = torch.softmax(B, dim=-1)
            S = torch.einsum('bio,bioh->boh', C, U)
            V = squash(S, dim=-1)
            B = B + torch.einsum('bioh,boh->bio', U, V)
        return V


# Encoder part construction
class Encoder(nn.Module):
    def __init__(self, in_h, in_w, in_c,
                 n_primary_caps_groups, primary_caps_dim,
                 n_digit_caps, digit_caps_dim,
                 ):
        super().__init__()
        self.image2primary_capsules = nn.Sequential(
            nn.Conv2d(in_c, 256, kernel_size=9),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, n_primary_caps_groups * primary_caps_dim, kernel_size=9, stride=2),
            Rearrange('b (caps hid) h w -> b (h w caps) hid', caps=n_primary_caps_groups, hid=primary_caps_dim),
        )
        _, n_primary_capsules, _ = self.image2primary_capsules(torch.zeros(1, in_c, in_h, in_w)).shape
        self.primary2digit_capsules = CapsuleLayerWithRouting(
            in_caps=n_primary_capsules, in_hid=primary_caps_dim,
            out_caps=n_digit_caps, out_hid=digit_caps_dim,
        )
    def forward(self, images, routing_iterations=routing_iter):
        primary_capsules = self.image2primary_capsules(images) * 0.01
        return self.primary2digit_capsules(primary_capsules, routing_iterations)


# Decoder part construction
def Decoder(n_caps, caps_dim, output_h, output_w, output_channels):
    return nn.Sequential(
        Mix('b caps caps_dim -> b hidden', weight_shape='caps caps_dim hidden', caps=n_caps, caps_dim=caps_dim, hidden=512),
        nn.ReLU(inplace=True),
        nn.Linear(512, 1024),
        nn.ReLU(inplace=True),
        Mix('b hidden -> b c h w', weight_shape='hidden c h w', hidden=1024, h=output_h, w=output_w, c=output_channels),
        nn.Sigmoid(),
    )


# Margin loss function
def margin_loss(class_capsules, target_one_hot, m_minus=0.1, m_plus=0.9, loss_lambda=0.5):
    caps_norms = torch.norm(class_capsules, dim=2)
    assert caps_norms.max() <= 1.001, 'capsules outputs should be bound by unit norm'
    loss_sig = torch.clamp(m_plus - caps_norms, 0) ** 2
    loss_bkg = torch.clamp(caps_norms - m_minus, 0) ** 2
    loss = target_one_hot * loss_sig + loss_lambda * (1.0 - target_one_hot) * loss_bkg
    return reduce(loss, 'b cls -> b', 'sum')


# Load MNIST dataset
def load_mnist(batch_size, workers=0):
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=(0, 0), translate=(0.1, 0.1), fill=0),
        transforms.ToTensor(),
    ])
    test_transform = transforms.ToTensor()
    training_data_loader = DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=train_transform),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    testing_data_loader = DataLoader(
        datasets.MNIST('./data', train=False, download=True, transform=test_transform),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    image_shape = (28, 28, 1)
    n_classes = 10
    return training_data_loader, testing_data_loader, image_shape, n_classes


# Build the model
## Data loader
train_loader, test_loader, (image_h, image_w, image_c), n_classes = load_mnist(batch_size=batch_size, workers=0)

## Encoder part
encoder = Encoder(
    in_h=image_h, in_w=image_w, in_c=image_c,
    n_primary_caps_groups=32, primary_caps_dim=8,
    n_digit_caps=n_classes, digit_caps_dim=16,
).to(device)

## Decoder part
decoder = Decoder(
    n_caps=n_classes, caps_dim=16,
    output_h=image_h, output_w=image_w, output_channels=image_c,
).to(device)

## Optimizer
optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()])


# Train the model
print("---------------------------------")
print("Training process initiated")
print("---------------------------------")
for epoch in range(epoch_num):
    print("++++++++++++++++")
    print("Epoch {}/{}:".format(epoch, epoch_num))
    for i, (images, labels) in enumerate(train_loader):
        digit_capsules = encoder(images.to(device))
        labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=10).float().to(device)
        loss = margin_loss(digit_capsules, labels_one_hot).mean()
        reconstructed = decoder(digit_capsules * rearrange(labels_one_hot, 'b caps -> b caps 1'))
        reconstruction_loss_mse = (images.to(device) - reconstructed).pow(2).mean()
        loss += reconstruction_loss_mse * 10.
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i%10 == 0:
            print("{}/{} iterations done".format(i, len(train_loader)))
        else:
            pass
    print("Testing")
    accuracies = []
    for images, labels in test_loader:
        digit_capsules = encoder(images.to(device)).cpu()
        predicted_labels = digit_capsules.norm(dim=2).argmax(dim=1)
        accuracies += asnumpy(predicted_labels == labels).tolist()
    print("+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-")
    print(f'epoch {epoch} accuracy: {np.mean(accuracies)}')
