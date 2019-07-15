from PIL import Image
import os
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

src_path = "dataset/"
dst_path = "dataset/"
resize_multiple_images(src_path, dst_path)
