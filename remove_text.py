import matplotlib.pyplot as plt
import cv2
import math
import numpy as np
from craft_text_detector import (
    read_image,
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
    export_detected_regions,
    export_extra_results,
    empty_cuda_cache
)

import warnings
warnings.filterwarnings('ignore')



# load models
refine_net = load_refinenet_model(cuda=True)
craft_net = load_craftnet_model(cuda=True)

# set image path and export folder directory
# image = '/home/bis/Projects/Classes/Deeplearning_class/project/data/img/01235.png' # can be filepath, PIL image or numpy array


def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2)/2)
    y_mid = int((y1 + y2)/2)
    return (x_mid, y_mid)


def inpaint_text(img_path):
    # read image
    output_dir = 'outputs/'

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
        img = cv2.inpaint(image, mask, 7, cv2.INPAINT_NS)
                 
    return(img)


if __name__ == "__main__":
    #https://towardsdatascience.com/remove-text-from-images-using-cv2-and-keras-ocr-24e7612ae4f4
    inpaint_text("/home/bis/Projects/Classes/Deeplearning_class/project/data/img/01235.png")

    # unload models from gpu
    empty_cuda_cache()