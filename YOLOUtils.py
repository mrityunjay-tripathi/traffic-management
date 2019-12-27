import torch, torchvision
import os, numpy as np
import pandas as pd
from PIL import Image
import argparse, optparse
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from sklearn.cluster import KMeans


### Get information of the the true bounding boxes
### DONE
def TrueBoxes(xml_file_path):
    root = ET.parse(xml_file_path).getroot()
    true_boxes = []
    class_label = {'plate':0}
    for member in root.findall('object'):
        IMG_W = int(root.find('size')[0].text)
        IMG_H = int(root.find('size')[1].text)
        # path = root.find('path').text
        value = (IMG_W,
                IMG_H,
                int(member[4][0].text), #xmin
                int(member[4][1].text), #ymin
                int(member[4][2].text), #xmax
                int(member[4][3].text), #ymax
                class_label[member[0].text]) #class
        true_boxes.append(value)
    
    return true_boxes

### Get normalized width, height and center=(x,y) of the bounding boxes of an image
def ProcessedBB(boxes, anchors, image_shape, grid_shape, num_of_classes):
    num_anchors = len(anchors)
    grid = np.zeros(shape=(grid_shape[0], grid_shape[1], num_anchors, 6+num_of_classes))
    reduction_factor = image_shape[0]/grid_shape[0]
    
    for box in boxes:
        box_center = ((box[4]+box[2])/2, (box[5]+box[3])/2)
        grid_x = int(box_center[0]//reduction_factor)
        grid_y = int(box_center[1]//reduction_factor)
        # best IOU of true box and anchor boxes.
        x, y = box_center[0]/box[0], box_center[1]/box[1]
        w, h = abs(box[4]-box[2])/box[0], abs(box[5]-box[3])/box[1]
        box_area = w*h
        best_iou = 0
        best_anchor = 0
        for i, anchor in enumerate(anchors):
            anchor_area = anchor[0]*anchor[1]
            intersection_area = min(h,anchor[1])*min(w,anchor[0])
            iou = intersection_area/(box_area + anchor_area - intersection_area)
            if iou>best_iou:
                best_iou = iou
                best_anchor = i
        grid[grid_x, grid_y, best_anchor, 0:6] = np.array([1, best_iou, x, y, w, h])
        grid[grid_x, grid_y, best_anchor, 6:] = np.eye(num_of_classes)[box[-1]]
    return grid

def Loss(batch_output, batch_actual, anchors, num_of_classes):
    '''
    Arguments:
    batch_output :- 
        - type = tensor
        - output from yolo architecture
        - shape = (batch_size, grid_shape[0], grid_shape[1], num_of_anchors*(5 + num_of_classes))
    batch_actual :-
        - type = tensor
        - preprocessed grid information of bounding boxes
        - shape = (batch_size, grid_shape[0], grid_shape[1], num_of_anchors, 7)
    
    Return:
    loss    
    '''
    L_coord = 5
    L_noobj = 0.5
    pred_xy, pred_wh, pred_confidence, pred_class_probs = Head(batch_output = batch_output, 
                                                           anchors = anchors, 
                                                           num_of_classes = num_of_classes)
    
    pass

### extract (x,y) and width-height of predicted bounding boxes.
### ...and box_confidence, box class probabilities
### DONE
def Head(batch_output, anchor_boxes, num_of_classes):

    num_anchors = len(anchor_boxes)

    batch_output_reshaped = batch_output.view(batch_output.shape[0], #num_batches
                                              batch_output.shape[1], #height
                                              batch_output.shape[2], #width
                                              num_anchors,           #number of anchor boxes
                                              batch_output.shape[3]//num_anchors) #channels//num_anchors
    box_confidence = torch.sigmoid(batch_output_reshaped[..., 0].unsqueeze(-1))
    box_class_probs = torch.softmax(batch_output_reshaped[...,5:], dim = -1)
    box_xy = torch.sigmoid(batch_output_reshaped[...,1:3])
    box_wh = torch.exp(batch_output_reshaped[...,3:5])
    return box_xy, box_wh, box_confidence, box_class_probs
    

def FilterBoxes(box_confidence, box_class_probs, boxes,
             score_threshold = 0.5):
    
    box_scores = torch.mul(box_confidence, box_class_probs)
    box_classes = torch.argmax(box_scores, axis = -1)
    box_class_scores = torch.max(box_scores, axis = -1).values

    filtering_mask = box_class_scores>=score_threshold
    scores = box_class_scores[filtering_mask]
    boxes = boxes[filtering_mask]
    classes = box_classes[filtering_mask]
    return scores, boxes, classes