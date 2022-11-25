import os.path as osp

import torch
from torch.utils.data import Dataset
import cv2


class RellisImage(Dataset):
    def __map2tensor(label_map):
        result = torch.zeros(max(label_map.keys()) + 1)
        for key, value in label_map.items():
            result[key] = value
        return result.long()

    label_map = {
        0: 0, 1: 0, 3: 1, 4: 2, 5: 3,
        6: 4, 7: 5, 8: 6, 9: 7, 10: 8,
        12: 9, 15: 10, 17: 11, 18: 12, 19: 13,
        23: 14, 27: 15, 29: 1, 30: 1, 31: 16,
        32: 4, 33: 17, 34: 18
    }
    label_inverse_map = {v: k for k, v in tuple(label_map.items())[::-1]}
    label_map = __map2tensor(label_map)
    label_inverse_map = __map2tensor(label_inverse_map)
    
    class_names = [
        "void", "grass", "tree", "pole", "water", 
        "sky", "vehicle", "object", "asphalt", "building", 
        "log", "person", "fence", "bush", "concrete", 
        "barrier", "puddle", "mud", "rubble"
    ]
    num_classes = len(label_inverse_map)
    
    def __init__(self, root, split, transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        
        if split not in ('train', 'test', 'val'):
            raise RuntimeError(f'{split} dataset is not supported.')
        
        list_filename = split + '.lst'
        self.filenames = []
        with open(osp.join(self.root, list_filename)) as file:
            lines = file.readlines()
            for line in lines:
                img, label = line.strip().split(' ')
                img = osp.join(self.root, img)
                label = osp.join(self.root, label)
                
                if not osp.exists(img):
                    raise FileNotFoundError(f"Cannot find the input image: '{img}'.")
                if not osp.exists(label):
                    raise FileNotFoundError(f"Cannot find the label image: '{label}'.")
                
                self.filenames.append((img, label))
        print(f'Rellis-3D_{self.split}: found {len(self.filenames)} samples.')
        
        self.load_x = RellisImage.__read_image
        self.load_y = RellisImage.__read_image
            
    def __getitem__(self, idx):
        x_filename, y_filename = self.filenames[idx]
        x = self.load_x(x_filename)
        y = self.load_y(y_filename)
        
        if self.transform is not None:
            x, y = self.transform(image=x, mask=y).values()
        return x, y
    
    def __len__(self):
        return len(self.filenames)
    
    @staticmethod
    def __read_image(filename):
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
