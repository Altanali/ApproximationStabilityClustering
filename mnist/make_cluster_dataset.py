import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

from model import Autoencoder
from data import VisionDataset


def get_bottleneck_features(dataloader, model, classes, num_samples_per_class):
    bottleneck_features = []
    labels = []

    class_samples_count = {key: 0 for key in classes}

    with torch.no_grad():
        for data, target in dataloader:
            # Check if we have enough samples for each class
            if all(
                count >= num_samples_per_class for count in class_samples_count.values()
            ):
                break

            data = data.flatten(1, -1).to(device)
            # Get bottleneck features
            _, h = model(data)
            h = h.detach().cpu()

            for feature, label in zip(h, target):
                if (
                    label.item() in class_samples_count
                    and class_samples_count[label.item()] < num_samples_per_class
                ):
                    bottleneck_features.append(feature.numpy())
                    labels.append(label.item())
                    class_samples_count[label.item()] += 1

    return np.array(bottleneck_features), np.array(labels)


dataset = "mnist"
data_dir = "./datasets/MNIST"
save_dir = "./out"
batch_size = 128
model_weights_path = "./out/mnist_ae_ep=1000.pt"
num_workers = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_samples = 100  # number of samples per class to save
classes = (0, 1, 2)  # classes to save

# Load the model
model = Autoencoder()
model.load_state_dict(torch.load(model_weights_path, map_location=device))
model.to(device)
model.eval()

# DataLoader
vision_dataset = VisionDataset(name=dataset, root=data_dir)
train_loader, _ = vision_dataset.get_dataloaders(
    batch_size=batch_size, num_workers=num_workers
)

features, labels = get_bottleneck_features(train_loader, model, classes, num_samples)

# Save the dataset
np.save(f"{save_dir}/bottleneck_features.npy", features)
np.save(f"{save_dir}/labels.npy", labels)
