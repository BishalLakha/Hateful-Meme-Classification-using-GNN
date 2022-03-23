import matplotlib.pyplot as plt
import cv2
import os
import math
import numpy as np
from tqdm import tqdm
import time
from craft_text_detector import (
    read_image,
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
    empty_cuda_cache
)

import warnings
warnings.filterwarnings('ignore')



def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2)/2)
    y_mid = int((y1 + y2)/2)
    return (x_mid, y_mid)


def inpaint_text(img_path,craft_net=None, refine_net=None):
    # read image
    output_dir = './preprocessed_images/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    
    img_name = os.path.basename(img_path)
    img_save_path = os.path.join(output_dir, img_name)

    if os.path.exists(img_save_path):
        return None

    image = read_image(img_path)


    # perform prediction
    prediction_result = get_prediction(
        image=image,
        craft_net=craft_net,
        refine_net=refine_net,
        text_threshold=0.7,
        link_threshold=0.4,
        low_text=0.4,
        cuda=True,
        long_size=1280
    )


    # start = time.time()

    prediction_boxes = prediction_result["boxes"]
    mask = np.zeros(image.shape[:2], dtype="uint8")

    for i in range(len(prediction_boxes)):
        box = prediction_boxes[i]
        x0, y0 = box[0]
        x1, y1 = box[1] 
        x2, y2 = box[2]
        x3, y3 = box[3] 
        
        x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
        x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)
        
        thickness = int(math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 ))
        
        cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255,    
        thickness)
        img = cv2.inpaint(image, mask, 4,cv2.INPAINT_TELEA)

    cv2.imwrite(img_save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # print(time.time()-start)

                 
    # print('Image saved to: ', img_save_path)

def preprocess_memes(data_dir):
    """Remove text from images and save them to a new directory
    """
    # load models
    refine_net = load_refinenet_model(cuda=True)
    craft_net = load_craftnet_model(cuda=True)

    print("Model loaded")

    for img_path in tqdm(os.listdir(data_dir)):
        if img_path.endswith('.png'):
            inpaint_text(os.path.join(data_dir, img_path), craft_net, refine_net)
    
    # Remove models from GPU
    empty_cuda_cache()

if __name__ == "__main__":
    #https://towardsdatascience.com/remove-text-from-images-using-cv2-and-keras-ocr-24e7612ae4f4

    preprocess_memes("./data/img/")
