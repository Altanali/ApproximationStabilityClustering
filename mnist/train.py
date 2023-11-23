import os
from PIL import Image

import numpy as np
import torch
import torch.nn as nn

from model import Autoencoder
from data import VisionDataset


def save_images(orig, recon, shape=(28, 28), filename="comparison.png"):
    # Reshape the flat images
    orig = orig.squeeze().reshape(shape).detach().cpu().numpy()
    recon = recon.squeeze().reshape(shape).detach().cpu().numpy()

    # Combine the images side-by-side
    stack_img = np.hstack((orig, recon))

    # Save using PIL
    img_to_save = Image.fromarray(np.uint8(stack_img * 255), 'L')  # 'L' mode for grayscale
    img_to_save.save(f"{filename}")


dataset     = 'mnist'
data_dir    = '/mnt/datasets/MNIST'
save_dir    = './out'
batch_size  = 128
hidden_size = 2
lr          = 2
wd          = 1e-5
momentum    = 0.9
num_epochs  = 1000
num_workers = 4
device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = Autoencoder(hidden_size=hidden_size)
model = model.to(device)
vision_dataset = VisionDataset(name=dataset, root=data_dir)
train_loader, test_loader = vision_dataset.get_dataloaders(batch_size=batch_size, num_workers=num_workers)

optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=momentum, nesterov=True)
criterion = nn.MSELoss()

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    os.makedirs(f'{save_dir}/img')

model.train()
for epoch in range(1, num_epochs+1):
    sum_loss = 0
    for i, (x, _) in enumerate(train_loader):
        x = x.flatten(1, -1).to(device)
        optimizer.zero_grad()
        x_hat, _ = model(x)
        loss = criterion(x_hat, x)
        loss.backward()
        sum_loss += loss.item()
        optimizer.step()
    scheduler.step()
    print(f'Epoch {epoch}/{num_epochs}, Average Loss={sum_loss / len(train_loader):.4f}')
    save_images(x[0], x_hat[0], filename=f'{save_dir}/img/epoch_{epoch}.png')

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

torch.save(model.state_dict(), f'./{save_dir}/{dataset}_ae_ep={num_epochs}.pt')
