import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import utils

path_to_train_images = '/home/mrityunjay/Documents/Object Detection - Project/dataset/train_images/'
path_to_test_images = '/home/mrityunjay/Documents/Object Detection - Project/dataset/test_images/'
path_to_xml = '/home/mrityunjay/Documents/Object Detection - Project/dataset/xml_details/'

def get_normalized_dimensions(path_to_xml):
    data = utils.xml_to_csv(path_to_xml)
    IMG_W = data['width'][0]
    IMG_H = data['height'][0]
    data['x'] = (data['xmax'] + data['xmin'])/2/IMG_W
    data['y'] = (data['ymax'] + data['ymin'])/2/IMG_H
    data['w'] = (data['xmax'] - data['xmin'])/IMG_W
    data['h'] = (data['ymax'] - data['ymin'])/IMG_H

    return data

def get_boxes(data, num_of_boxes):

    rect = data[['x', 'h']].values
    kmeans = KMeans(n_clusters = num_of_boxes, random_state = 0).fit(rect)
    #labels = kmeans.labels_
    boxes_dims = kmeans.cluster_centers_
    return boxes_dims


def plot_boxes(boxes_dims):
    center = (0, 0)
    for i in boxes_dims:
        xmin = center[0] - i[0]/2
        xmax = center[0] + i[0]/2
        ymin = center[1] - i[1]/2
        ymax = center[1] + i[1]/2
        plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin])
    plt.title('Anchor Boxes')
    plt.show()


data = get_normalized_dimensions(path_to_xml)
boxes_dims = get_boxes(data, num_of_boxes = 5)
print(boxes_dims)
plot_boxes(boxes_dims)
