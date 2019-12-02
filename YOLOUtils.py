import torch, torchvision
import os, numpy as np
import pandas as pd
from PIL import Image
import argparse, optparse
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from sklearn.cluster import KMeans


### Get information of the the bounding boxes
def BoundingBoxes(xml_file_path, format = 'numpy'):
    root = ET.parse(xml_file_path).getroot()
    bb_info = {'path':[],'width':[], 'height':[], 
               'class':[], 'xmin':[], 'ymin':[], 'xmax':[], 'ymax':[]}

    for member in root.findall('object'):
        bb_info['path'].append(root.find('path').text)
        bb_info['width'].append(int(root.find('size')[0].text))
        bb_info['height'].append(int(root.find('size')[1].text))
        bb_info['class'].append(member[0].text)
        bb_info['xmin'].append(int(member[4][0].text))
        bb_info['ymin'].append(int(member[4][1].text))
        bb_info['xmax'].append(int(member[4][2].text))
        bb_info['ymax'].append(int(member[4][3].text))
    
    if format=='numpy':
        return pd.DataFrame.from_dict(bb_info).values
    elif format=='dataframe':
        return pd.DataFrame.from_dict(bb_info)

### Get normalized width, height and center=(x,y) of the bounding boxes of an image
def NormalizedBB(xml_file_path):
    root = ET.parse(xml_file_path).getroot()
    bb_norm = {'path':[], 'class':[], 'x':[], 'y':[], 'w':[], 'h':[]}
    for member in root.findall('object'):
        IMG_W = int(root.find('size')[0].text)
        IMG_H = int(root.find('size')[1].text)
        bb_norm['path'].append(root.find('path').text)
        bb_norm['class'].append(member[0].text)
        bb_norm['x'].append((int(member[4][2].text) + int(member[4][0].text))/2/IMG_W)
        bb_norm['y'].append((int(member[4][3].text) + int(member[4][1].text))/2/IMG_H)
        bb_norm['w'].append((int(member[4][2].text) - int(member[4][0].text))/IMG_W)
        bb_norm['h'].append((int(member[4][3].text) - int(member[4][1].text))/IMG_H)
    return bb_norm

### Make n number of anchor boxes 
def GetAnchorBoxes(annotations_path, num_of_boxes = 5):
    rect = []

    for filename in os.listdir(annotations_path):
        w = NormalizedBB(annotations_path + filename)['w']
        h = NormalizedBB(annotations_path + filename)['h']
        rect.append([w,h])
    
    rect = np.squeeze(np.array(rect))
    kmeans = KMeans(n_clusters = num_of_boxes, random_state = 0).fit(rect)
    #labels = kmeans.labels_
    boxes = kmeans.cluster_centers_
    return boxes


### Visualize the anchor boxes
def PlotAnchorBoxes(boxes):
    center = (0, 0)
    for i in boxes:
        xmin = center[0] - i[0]/2
        xmax = center[0] + i[0]/2
        ymin = center[1] - i[1]/2
        ymax = center[1] + i[1]/2
        plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin])
    plt.title('Anchor Boxes')
    plt.show()


### convert normalized anchor boxes back to bounding boxes with coordinates
def BoxCoordinates(box, image_shape = (480,640)):
    ### x = (xmin + xmax)/2/IMG_WIDTH
    ### y = (ymin + ymax)/2/IMG_HEIGHT
    ### w = (xmax - xmin)/IMG_WIDTH
    ### h = (ymax - ymin)/IMG_WIDTH
    x,y,w,h = box.T
    xmax = image_shape[1]*(2*x + w).unsqueeze(-1)
    xmin = image_shape[1]*(2*x - w).unsqueeze(-1)
    ymax = image_shape[0]*(2*y + h).unsqueeze(-1)
    ymin = image_shape[0]*(2*y - w).unsqueeze(-1)

    coordinates = torch.cat((xmax,xmin,ymax,ymin), axis = -1)
    return coordinates


### Intersection over union ratio
def IoU(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection_area = (x2 - x1)*(y2 - y1)

    box1_area = (box1[3] - box1[1])*(box1[2] - box1[0])
    box2_area = (box2[3] - box2[1])*(box2[2] - box2[0])

    union_area = (box1_area + box2_area) - intersection_area

    iou = torch.div(intersection_area, union_area)
    return iou.item()


### Non Max Suppression
def NMS(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):

    nms_indices = torchvision.ops.nms(boxes, scores, iou_threshold)
    scores = scores[nms_indices]
    boxes = boxes[nms_indices]
    classes = classes[nms_indices]

    return scores, boxes, classes


### YOLO loss function
def cost(output, box_confidence, actual, ):
    L_coord = 5
    L_noobj = 0.5
    '''
    Required:
    box_confidence, (x,y), (h,w), (x_hat, y_hat), (h_hat, w_hat), actual classes, output classes
    '''
    cost = L_coord*()
    

def Evaluate(output, image_shape = (480.0, 640.0),
             num_of_classes = 10, num_of_anchor_boxes = 5,
             score_threshold = 0.6, iou_threshold = 0.5):
    
    output_reshaped = output.view(output.shape[0], 
                                  output.shape[1], 
                                  num_of_anchor_boxes, 
                                  output.shape[2]//num_of_anchor_boxes)
    
    print(output_reshaped.shape)
    box_confidence = output_reshaped[:,:,:,0].unsqueeze(-1)
    print(box_confidence.shape)
    box_class_probs = output_reshaped[:,:,:,5:5+num_of_classes]
    print(box_class_probs.shape)
    boxes = output_reshaped[:,:,:,1:5]
    box_scores = torch.mul(box_confidence, box_class_probs)
    print(box_scores.shape)
    box_classes = torch.argmax(box_scores, axis = -1)
    box_class_scores = torch.max(box_scores, axis = -1).values

    filtering_mask = box_class_scores>=score_threshold
    scores = box_class_scores[filtering_mask]
    boxes = boxes[filtering_mask]
    classes = box_classes[filtering_mask]
    return scores, boxes, classes
    

# output = torch.rand(12,16,125)
# scores, boxes, classes = Evaluate(output, num_of_classes=20)

# coordinates = BoxCoordinates(boxes)
# print(coordinates)
 
    
