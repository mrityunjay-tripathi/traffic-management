import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import tensorflow as tf

def get_boxes(data, num_of_boxes):

    rect = data[['w', 'h']].values
    kmeans = KMeans(n_clusters = num_of_boxes, random_state = 0).fit(rect)
    #labels = kmeans.labels_
    boxes = kmeans.cluster_centers_
    return boxes


def plot_boxes(boxes):
    center = (0, 0)
    for i in boxes:
        xmin = center[0] - i[0]/2
        xmax = center[0] + i[0]/2
        ymin = center[1] - i[1]/2
        ymax = center[1] + i[1]/2
        plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin])
    plt.title('Anchor Boxes')
    plt.show()



def non_max_suppression(boxes, scores, classes, max_output_size, iou_threshold = 0.6):
    """
    Arguments :
    -----------------------------------------------------------------------------
    boxes : It is a 2D array of the format -
            array([(x1, y1, w1, h1),
                   (x2, y2, w2, h2),
                   ....  ....  ....])
            with shape (num_of_boxes, 4)
            
    scores : It is a 1D array having scores for the corresponding bounding boxes.
             Eg:- array([[s1], [s2], ....   ..   ....])
             
    max_output_size : The number of boxes we want to output.

    iou_threshold : A float representing the threshold for deciding
                    whether boxes overlap too much with respect to IOU.

    Returns :
    ------------------------------------------------------------------------------
    boxes : 2D array of shape (max_output_size, 4)

    scores : 1D array of shape (max_output_size, )

    classes : 1D array of shape (max_output_size, )

    """
    # For documentation of module tf.image.non_max_suppression
    # visit https://www.tensorflow.org/api_docs/python/tf/image/non_max_suppression
    
    nms_indices = tf.image.non_max_suppression(boxes = boxes,
                                               scores = scores,
                                               max_output_size = max_output_size,
                                               iou_threshold = iou_threshold)

    # For more info about tf.gather
    # visit https://www.tensorflow.org/api_docs/python/tf/gather
    
    boxes = tf.gather(boxes, nms_indices)
    scores = tf.gather(scores, nms_indices)
    classes = tf.gather(classes, nms_indices)

    return boxes, scores, classes




















