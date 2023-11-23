from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

class VisionDataset:
    def __init__(self, name, root, download=True):
        self.root = root
        self.download = download

        if name.lower() == 'mnist':
            self.dataset_builder = self.get_mnist_dataset
        else:
            raise NotImplementedError('Dataset {} not implemented'.format(name))

    def get_dataloaders(self, batch_size=32, num_workers=4):
        train_dataset, test_dataset = self.dataset_builder()
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        return train_loader, test_loader

    def get_mnist_dataset(self):
        train_dataset = MNIST(
            self.root,
            train=True,
            download=self.download,
            transform = transforms.Compose([
                transforms.ToTensor(),
                #transforms.Normalize((0.1307,), (0.3081,))
            ]),
        )

        test_dataset = MNIST(
            self.root,
            train=False,
            download=self.download,
            transform = transforms.Compose([
                transforms.ToTensor(),
                #transforms.Normalize((0.1307,), (0.3081,))
            ]),
        )

        return train_dataset, test_dataset
