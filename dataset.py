import os, torch
from PIL import Image
from torch.utils.data import Dataset
from YOLOUtils import ProcessedBB, TrueBoxes

class NumberPlateDataset(Dataset):

    def __init__(self, 
                 train_images_path, 
                 annotations_path,
                 anchors,
                 transform = transforms.Compose([transforms.ToTensor()])):
        self.train_images_path = train_images_path
        self.annotations_path = annotations_path
        self.transform = transform
        self.class_dict = {"license plate":1}
        self.image_shape = (480, 640)
        self.grid_shape = (12, 16)
        self.num_of_classes = 1
        self.anchors = anchors
    
    def files(self):
        f = []
        for filename in os.listdir(self.annotations_path):
            f.append(filename)
        return f

    def __len__(self):
        return len(self.files())
    

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = torch.tolist()
        f = self.files()
        xml_path = f[idx]
        boxes = TrueBoxes(os.path.join(self.annotations_path, xml_path))
        true_boxes = ProcessedBB(boxes=boxes,
                                 anchors = self.anchors,
                                 image_shape = self.image_shape,
                                 grid_shape = self.grid_shape,
                                 num_of_classes = self.num_of_classes)
        img = Image.open(self.train_images_path + f[idx][:-3] + 'jpg')

        if self.transform:
            img = self.transform(img)
        return {'image':img, 'true_boxes':true_boxes}