import os
import pandas as pd
import xml.etree.ElementTree as ET
from PIL import Image

def rename_multiple_files(path):

    i=0

    for filename in os.listdir(path):
        try:
            f,extension = os.path.splitext(path+filename)
            src=path+filename
            dst=path+str(i)+extension
            os.rename(src,dst)
            i+=1
            print('Rename successful.')
        except:
            i+=1
            print('Unable to rename.')


            
def resize_multiple_images(src_path, dst_path):
    # Here path is the location where images are saved.
    for filename in os.listdir(src_path):
        try:
            img=Image.open(src_path+filename)
            new_img = img.resize((832,624))
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
            new_img.save(dst_path+filename)
            print('Resized and saved {} successfully.'.format(filename))
        except:
            continue

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
    columns = ['filename',
	       'width', 
	       'height', 
	       'class', 
               'xmin', 
	       'ymin', 
   	       'xmax',
	       'ymax']
    xml_df = pd.DataFrame(xml_list, columns = columns)
    return xml_df



