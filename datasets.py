import glob
import random
import os
import csv

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

class InverseTransformImageDataset(Dataset):

     """
         image = input_image -> pre_transform -> post_transform
         target = input_image -> pre_transform -> target_transforms -> post_transform
     """

    def __init__(self, root, pre_transforms=None, target_transforms=None, post_transform=None, mode='train'):
        self.pre_transform = transforms.Compose(transforms)
        self.target_transform = transforms.Compose(transforms)
        self.post_transform = transforms.Compose(transforms)
        self.unaligned = unaligned

        self.files = sorted(glob.glob(os.path.join(root, '%s' % mode) + '/*.*'))

    def __getitem__(self, index):
        image = self.pre_transform(Image.open(self.files[index % len(self.files)]))
        target = self.target_transform(image)

        image = self.post_transform(image)
        target = self.post_transform(target)

        return {'A': image, 'B': target}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

class CelebA(Dataset):
    """"

    Possible attributes:
        5_o_Clock_Shadow
        Arched_Eyebrows
        Attractive
        Bags_Under_Eyes
        Bald Bangs
        Big_Lips
        Big_Nose
        Black_Hair
        Blond_Hair
        Blurry Brown_Hair
        Bushy_Eyebrows Chubby
        Double_Chin
        Eyeglasses
        Goatee
        Gray_Hair
        Heavy_Makeup
        High_Cheekbones
        Male
        Mouth_Slightly_Open
        Mustache
        Narrow_Eyes
        No_Beard
        Oval_Face Pale_Skin
        Pointy_Nose
        Receding_Hairline
        Rosy_Cheeks
        Sideburns Smiling
        Straight_Hair
        Wavy_Hair
        Wearing_Earrings
        Wearing_Hat
        Wearing_Lipstick
        Wearing_Necklace
        Wearing_Necktie
        Young
    """

    def __init__(self, root, transforms_=None, unaligned=False, attribute = "Eyeglasses"):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        #self.files = sorted(glob.glob(os.path.join(root, 'img_align_celeba') + '/*.*'))
        attribute_file = os.path.join(root, 'list_attr_celeba.txt')
        image_path = os.path.join(root, 'img_align_celeba')

        self.files_A = []
        self.files_B = []

        with open(attribute_file, "r") as f:
            reader = csv.reader(f, delimiter=" ")
            num_files = next(reader, None)
            attribute_list = next(reader, None)
            attribute_list = [x.lower() for x in attribute_list]
            attribute = attribute.lower()
            if attribute.lower() in attribute_list:
                attribute_idx = attribute_list.index(attribute)

                for row in reader:
                    row = [x for x in row if x != '']
                    if row[attribute_idx+1] == '1':
                        self.files_A.append(os.path.join(image_path,row[0]))
                    else:
                        self.files_B.append(os.path.join(image_path,row[0]))
            #import pdb; pdb.set_trace()

    def __getitem__(self, index):
        #import pdb; pdb.set_trace()
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

if __name__ == "__main__":
    dataset = CelebA("/home/msu/Data/celeba", unaligned=True, attribute = "male")
