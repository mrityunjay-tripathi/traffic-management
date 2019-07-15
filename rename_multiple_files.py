import os

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

path="dataset/xml_details/"
rename_multiple_files(path)
