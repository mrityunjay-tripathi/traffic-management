import os, numpy as np
import torch, torchvision
from PIL import Image
import xml.etree.ElementTree as ET


def TrueBoxes(xml_file_path):
    """
    Arguments:
    1) xml_file_path - 
        # Location of xml file generated after annotating the image
        # type(xml_file_path) == <class 'str'>
    Returns:
    1) true_boxes - 
        # The extracted information about bounding boxes,
          such as height and width of image, coordinates of bounding boxes,
          classes of bounding boxes. It is a list of tuples.
        # type(true_boxes) == <class 'list'>
    """
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


def ProcessedBB(boxes, anchors, image_shape, grid_shape, num_classes):
    """
    Arguments:
    1) boxes -
        # The list of bounding boxes acquired from using 
          function TrueBoxes.
    2) anchors - 
        # the list of anchor boxes used.
        # eg. [(0.3,0.74), (0.5, 0.12), (0.47, 0.21)]
        # type(anchors) == <class 'list'>
    3) image_shape - 
        # shape of the image
    4) grid_shape -
        # shape of the final processed data from image i.e grid
        # aspect ratio to be maintained
        # eg. (19,19)
    5) num_classes - 
        # different types of classes of objects
        # type(num_classes) == <class 'int'>
    Returns:
    1) grid -
        # grid having information about each grid cell
    """
    num_anchors = len(anchors)
    grid = np.zeros(shape=(grid_shape[0], grid_shape[1], num_anchors, 6+num_classes))
    reduction_factor = image_shape[0]/grid_shape[0]
    
    for box in boxes:
        box_center = ((box[4]+box[2])/2, (box[5]+box[3])/2)
        grid_x = int(box_center[0]//reduction_factor)
        grid_y = int(box_center[1]//reduction_factor)
        # calculate intersection over union (iou)
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
        grid[grid_x, grid_y, best_anchor, 0:5] = np.array([1, x, y, w, h])
        grid[grid_x, grid_y, best_anchor, 5:] = np.eye(num_classes)[box[-1]]
    return grid


def Head(grid, num_anchors, true_grid = False):
    '''
    Arguments:
    1) grid -
        # grid obtained from ProcessedBB function
    2) num_anchors -
        # number of anchor boxes used
    3) true_grid -
        # type == bool
        # False if grid is output of yolo net.
        # True if grid is coming from from bounding boxes dataset.
    Returns:
    1) box_xy -
        # shape == (batch_size, grid_shape[0], grid_shape[1], num_anchors, 2)
        # axis = -1 contains center of each bounding boxes
    '''
    if true_grid:
        box_confidence = grid[..., 0].unsqueeze(-1)
        box_class_probs = grid[...,5:]
        box_xy = grid[...,1:3] # index 1 = x, index 2 = y
        box_wh = grid[...,3:5] # index 3 = w, index 4 = h
        return box_xy, box_wh, box_confidence, box_class_probs
    else:
        grid = grid.view(grid.shape[0], #num_batches
                                              grid.shape[1], #height
                                              grid.shape[2], #width
                                              num_anchors,           #number of anchor boxes
                                              grid.shape[3]//num_anchors) #channels//num_anchors
        box_confidence = torch.sigmoid(grid[..., 0].unsqueeze(-1))
        box_class_probs = torch.softmax(grid[...,5:], dim = -1)
        box_xy = torch.sigmoid(grid[...,1:3]) # index 1 = x, index 2 = y
        box_wh = torch.exp(grid[...,3:5]) # index 3 = w, index 4 = h
        return box_xy, box_wh, box_confidence, box_class_probs


def Loss(batch_output, batch_true):
    L_coord = 5
    L_noobj = 0.5
    num_anchors = batch_true.shape[-2]
    num_classes = batch_true.shape[-1] - 5
    pred_xy, pred_wh, pred_confidence, pred_class_probs = Head(grid = batch_output, 
                                                               num_anchors = num_anchors, 
                                                               true_grid = False)
    true_xy, true_wh, true_confidence, true_class_probs = Head(grid = batch_true, 
                                                               num_anchors = num_anchors, 
                                                               true_grid = True)

    # localization loss
    xy_loss = (pred_xy[0] - true_xy[0]).pow(2) + (pred_xy[1] - true_xy[1]).pow(2)
    wh_loss = (torch.sqrt(pred_wh[0]) - torch.sqrt(true_wh[0])).pow(2) + \
        (torch.sqrt(pred_wh[1]) - torch.sqrt(true_wh[1])).pow(2)
    localization_loss = L_coord*torch.sum(true_confidence*(xy_loss + wh_loss))

    # classification loss
    classification_loss = torch.sum(true_confidence*(pred_class_probs - true_class_probs).pow(2))

    # confidence loss
    intersect_wh = torch.max(torch.zeros_like(pred_wh), (pred_wh + true_wh)/2 - torch.abs(pred_xy - true_xy))
    intersection_area = intersect_wh[...,0]*intersect_wh[...,1]
    true_area = true_wh[...,0]*true_wh[...,1]
    pred_area = pred_wh[...,0]*pred_wh[...,1]
    iou = (intersection_area)/(true_area + pred_area - intersection_area)
    iou = iou.unsqueeze(-1)
    confidence_loss = torch.sum((true_confidence*iou - pred_confidence).pow(2)*true_confidence) 
    
    ### total loss
    loss = localization_loss + classification_loss + confidence_loss

    return loss


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