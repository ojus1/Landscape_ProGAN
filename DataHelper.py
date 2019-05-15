from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import os


img_shape = (256, 256)
default_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((img_shape[0], img_shape[1])),
        transforms.ToTensor(),
    ])

class LandscapeImages(Dataset):
    def __init__(self, root_dir="../../Datasets/landscapes/", transform=default_transform):
        self.root_dir = root_dir
        #self.imgs = glob(self.root_dir + "*")
        self.imgs = os.listdir(self.root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = Image.open(self.root_dir + self.imgs[index]).convert("RGB")
        img = self.transform(img)
        return img

if __name__ == "__main__":
    dataset = LandscapeImages()
    i = 0
    for x in dataset:
        print(x.shape, i)
        i += 1
