import torchvision
from torch.utils.data.dataset import Dataset


class MnistDataset(Dataset):
    r"""
    Thin wrapper around torchvision's built-in MNIST dataset.
    torchvision downloads the official MNIST files (from
    http://yann.lecun.com/exdb/mnist/, mirrored on torchvision's servers)
    into `im_path` the first time this class is instantiated, so no manual
    data preparation step is needed.
    """
    def __init__(self, split, im_path, im_size=28, im_channels=1, im_ext='png'):
        self.split = split
        self.im_size = im_size
        self.im_channels = im_channels

        # torchvision.datasets.MNIST only has 'train' and 'test' (=not train) splits
        is_train = split == 'train'
        self.data = torchvision.datasets.MNIST(root=im_path, train=is_train, download=True)
        print('Found {} images for split {}'.format(len(self.data), self.split))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        im, label = self.data[index]
        im_tensor = torchvision.transforms.ToTensor()(im)

        # Convert input to -1 to 1 range to match the generator's Tanh output.
        im_tensor = (2 * im_tensor) - 1
        return im_tensor
