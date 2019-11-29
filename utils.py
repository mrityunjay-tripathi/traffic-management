import os
import numpy as np
import pandas as pd
from PIL import Image
import argparse, optparse
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from sklearn.cluster import KMeans



def RenameMultipleFiles(path, prefix = ""):
    num = 0
    s = 0 # Number of files successfully renamed
    for filename in os.listdir(path):
        num+=1
        print(f"[#]Renaming: {filename} -> {prefix + str(num) + extension}", end = "\r")
        try:
            _, extension = os.path.splitext(path + filename)
            src = path + filename
            dst = path + prefix + str(num) + extension
            os.rename(src,dst)
            s += 1
        except Exception as e:
            print(f"Error: [{filename}]", e)
    print('[#]Process Complete!', 100*' ')
    print(f'{s}/{num} file(s) successfully renamed.')


            
def ResizeMultipleImages(src_path, dst_path, size):
    s = 0 # Number of files successfully resized.
    f = 0 # Number of files couldn't be resized.
    for filename in os.listdir(src_path):
        try:
            img = Image.open(src_path + filename)
            new_img = img.resize(size)
            print(f"[#]Resizing: {filename} from {img.size} to {new_img.size}", end = "\r")
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
            new_img.save(dst_path + filename)
            s += 1
        except Exception as e:
            print(f"Error: [{filename}]", e)
            f += 1
            continue
    print('Process Complete!',100*' ')
    print(f'{s}/{s + f} file(s) resized successfully.')


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


def PlotBoxes(boxes):
    center = (0, 0)
    for i in boxes:
        xmin = center[0] - i[0]/2
        xmax = center[0] + i[0]/2
        ymin = center[1] - i[1]/2
        ymax = center[1] + i[1]/2
        plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin])
    plt.title('Anchor Boxes')
    plt.show()


def NonMaxSuppression(boxes, scores, classes, max_output_size, iou_threshold = 0.6):    
    pass
    # TODO: Implement NMS in pytorch



if __name__ == "__main__":
    parser = optparse.OptionParser()
    parser.add_option("-p", "--path", dest = "action_path", 
                      help = "take action on all files in a folder")
    parser.add_option("-t", "--tag", dest = "tag", 
                      help = "prefix of names of files in a folder")
    parser.add_option("-x", "--lengthx", dest = "new_width",
                      help = "new width of all files in a folder")
    parser.add_option("-y", "--lengthy", dest = "new_height",
                      help = "new height of all files in a folder")

    (options, arguments) = parser.parse_args()
    path = options.action_path
    prefix = options.tag
    new_width = int(options.new_width)
    new_height = int(options.new_height)


    if prefix:
        RenameMultipleFiles(path, prefix = prefix)
    if new_width and new_height:
        ResizeMultipleImages(path, path, 
                             size = (new_width, new_height))
