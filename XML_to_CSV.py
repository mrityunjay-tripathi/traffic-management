import os
import pandas as pd
import xml.etree.ElementTree as ET


def convert(path):
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


df = convert('dataset/xml_details/')
df.to_csv('dataset/labelled_data.csv', index = False)
print(df)
