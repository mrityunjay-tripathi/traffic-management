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
    print('[#]Process Complete!',100*' ')
    print(f'{s}/{s + f} file(s) resized successfully.')


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
    elif new_width or new_height:
        edge = new_width if new_width else new_height
        ResizeMultipleImages(path, path, 
                             size = (edge, edge))
