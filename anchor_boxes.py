import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import utils


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


