import os, torch, glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from YOLOUtils import ProcessedBB, TrueBoxes

class NumberPlateDataset(Dataset):

    def __init__(self, 
                 train_images_path, 
                 annotations_path,
                 anchors,
                 transform):
        self.train_images_path = train_images_path
        self.annotations_path = annotations_path
        self.transform = transform
        self.image_shape = (480, 640)
        self.grid_shape = (12, 16)
        self.num_of_classes = 1
        self.anchors = anchors
        self.xml_files = sorted(glob.glob(f"{annotations_path}/*.xml"))
        self.image_files = sorted(glob.glob(f"{train_images_path}/*.jpg"))

    def __len__(self):
        return len(self.xml_files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = torch.tolist()
        boxes = TrueBoxes(self.xml_files[idx%len(self.xml_files)])
        true_boxes = ProcessedBB(boxes=boxes,
                                 anchors = self.anchors,
                                 image_shape = self.image_shape,
                                 grid_shape = self.grid_shape,
                                 num_classes = self.num_of_classes)
        img = Image.open(self.image_files[idx%len(self.image_files)])
        if self.transform:
            img = self.transform(img)
        return {'image':img, 'true_boxes':true_boxes}