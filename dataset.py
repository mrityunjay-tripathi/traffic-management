import os, torch, glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.yolo_utils import ProcessedBB, TrueBoxes

class NumberPlateDataset(Dataset):

    def __init__(self, 
                 images_path, 
                 labels_path,
                 anchors,
                 transform):
        self.images_path = images_path
        self.labels_path = labels_path
        self.transform = transform
        self.image_shape = (480, 640)
        self.grid_shape = (12, 16)
        self.num_of_classes = 1
        self.anchors = anchors
        self.image_files = sorted(glob.glob(f"{images_path}/*.jpg"))
        self.xml_files = sorted(glob.glob(f"{labels_path}/*.xml"))

    def __len__(self):
        return len(self.xml_files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = torch.tolist()
        boxes = TrueBoxes(self.xml_files[idx%len(self.xml_files)])
        true_boxes = ProcessedBB(boxes=boxes)
        img = Image.open(self.image_files[idx%len(self.image_files)])
        if self.transform:
            img = self.transform(img)
        
        noise = torch.rand(size = img.shape)/5
        noised_img = img + noise
        # To visualize the noised image, uncomment the below two lines
        # noised_img = transforms.Compose([transforms.ToPILImage()])(noised_img)
        # noised_img.show()
        return {'image':noised_img, 'true_boxes':true_boxes}
