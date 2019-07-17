import os
import argparse
import pandas as pd
import xml.etree.ElementTree as ET
from PIL import Image

def rename_multiple_files(path):
    name=0
    SUCCESSFUL = 0
    UNSUCCESSFUL = 0
    print('Processing...')
    for filename in os.listdir(path):
        try:
            f,extension = os.path.splitext(path+filename)
            src=path+filename
            dst=path+str(name)+extension
            os.rename(src,dst)
            name+=1
            SUCCESSFUL += 1
        except:
            name+=1
            UNSUCCESSFUL += 1
    print('Process Complete!')
    print('{} file(s) successfully resized.'.format(SUCCESSFUL))
    print('Unable to resize {} file(s).'.format(UNSUCCESSFUL))


            
def resize_multiple_images(src_path, dst_path, size):
    SUCCESSFUL = 0
    UNSUCCESSFUL = 0
    print('Processing...')
    for filename in os.listdir(src_path):
        try:
            img=Image.open(src_path+filename)
            new_img = img.resize(size)
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
            new_img.save(dst_path+filename)
            SUCCESSFUL += 1
        except:
            UNSUCCESSFUL += 1
            continue
    print('Process Complete!')
    print('{} file(s) successfully resized.'.format(SUCCESSFUL))
    print('Unable to resize {} file(s).'.format(UNSUCCESSFUL))


def xml_to_csv(path):
    xml_list = []
    for filename in os.listdir(path):
        root = ET.parse(path + filename).getroot()
        try:
            for member in root.findall('object'):
                value = (root.find('filename').text,
                         int(root.find('size')[0].text),
                         int(root.find('size')[1].text),
                         member[0].text,
                         int(member[4][0].text),
                         int(member[4][1].text),
                         int(member[4][2].text),
                         int(member[4][3].text))
                xml_list.append(value)
        except:
            continue
    columns = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns = columns)
    return xml_df



