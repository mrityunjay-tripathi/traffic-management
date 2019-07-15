import argparse, os
from PIL import Image
import tensorflow as tf
import numpy as np
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Lambda, Activation, Dense, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras import backend as K



class YOLO_object_detection:

    def __init__(self,):
        pass


    def filter_boxes(self, box_confidence, boxes, box_class_probs, threshold = 0.6):
        box_scores = box_confidence*box_class_probs
        box_classes = K.argmax(box_scores, axis = -1)
        box_class_scores = K.max(box_scores, axis = -1)

        filtering_mask = box_class_scores >= threshold

        boxes = tf.boolean_mask(boxrs, filtering_mask)
        classes = tf.boolean_mask(box_classes, filtering_mask)
        scores = tf.boolean_mask(box_class_scores, filtering_mask)

        return scores, boxes, classes

    def iou(self, box1, box2):
        xi1 = max(box1[0], box2[0])
        yi1 = max(box1[1], box2[1])
        xi2 = min(box1[2], box2[2])
        yi2 = min(box1[3], box2[3])
        inter_area = max(xi2 - xi1, 0)*max(yi2 - yi1, 0)

        box1_area = abs((box1[0] - box1[2])*(box1[1] - box1[3]))
        box2_area = abs((box2[0] - box2[2])*(box2[1] - box2[3]))
        union_area = box1_area + box2_area - inter_area

        iou = inter_area/union_area

        return iou

    def non_max_suppression(self, scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):
        max_boxes_tensor = K.variable(max_boxes, dtype='int32')
        K.get_session().run(tf.variables_initializer([max_boxes_tensor]))

        nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes_tensor)

        scores = K.gather(scores, nms_indices)
        boxes = K.gather(boxes, nms_indices)
        classes = K.gather(classes, nms_indices)

        return scores, boxes, classes
    
    def evaluate(self,):
        box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs

        boxes = yolo_boxes_to_corners(box_xy, box_wh)

        scores, boxes, classes = self.filter_boxes(box_confidence, boxes, box_class_probs, threshold = score_threshold)

        boxes = scale_boxes(boxes, image_shape)

        scores, boxes, classes = self.non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = iou_threshold)

        return scores, boxes, classes

    def predict(self, sess, image_file):
        image, image_data = preprocess_image("images/" + image_file, model_image_size = (608, 608))

        out_scores, out_boxes, out_classes = self.evaluate(yolo_outputs, image_shape = (720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5)

        print('Found {} boxes for {}'.format(len(out_boxes), image_file))
        
        colors = generate_colors(class_names)
        draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
        image.save(os.path.join("out", image_file), quality=90)
        output_image = scipy.misc.imread(os.path.join("out", image_file))
        imshow(output_image)
        
        return out_scores, out_boxes, out_classes
