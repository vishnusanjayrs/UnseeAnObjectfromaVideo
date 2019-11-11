import cv2
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt


def get_image_mask():
    # Root directory of the project
    ROOT_DIR = os.path.abspath(".")

    print(ROOT_DIR)

    # Import Mask RCNN
    sys.path.append(ROOT_DIR)  # To find local version of the library
    from mrcnn import utils
    import mrcnn.model as modellib
    from mrcnn import visualize
    # Import COCO config
    sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
    # import coco
    from samples.coco import coco


    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    # Directory of images to run detection on
    IMAGE_DIR = os.path.join(ROOT_DIR, "images")


    class InferenceConfig(coco.CocoConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    config.display()

     # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                   'bus', 'train', 'truck', 'boat', 'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                   'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                   'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                   'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                   'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear', 'hair drier', 'toothbrush']

    cap = cv2.VideoCapture('./VID_20190408_171217.mp4')
    frame_list = []

    img_dir = 'input/hallway'
    i = 0
    # loop through all the frames and store them in a list
    while (true):
        # Capture frame-by-frame
        i += 1
        ret, frame = cap.read()
        print(type(frame))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        print(i)
        frame_list.append(frame)
        # cv2.imshow('frame', frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame,(256,256))
        if (i > 0):
            image = frame

            plt.imsave('input/hallway_' + str(i) + '_input1.png', image)

            # Run detection
            results = model.detect([image], verbose=1)
            
            # Visualize results
            r = results[0]
            
            boxes = r['rois']
            classes = r['class_ids']
            n = boxes.shape[0]
            
            
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
            for j in range(n):
                if class_names[classes[j]] == 'person':
                    y1, x1, y2, x2 = boxes[j]
                    image[y1-3:y2+3, x1-3:x2+3, :] = 255
                    mask[y1-3:y2+3, x1-3:x2+3] = 255
            
            
            # plt.imshow(image)
            # plt.show()
            # plt.imshow(mask)
            # plt.show()
            plt.imsave('input/hallway_' + str(i) + '_input.png', image)
            plt.imsave('input/hallway_' + str(i) + '_mask.png', mask , cmap=matplotlib.cm.gray, vmin=0, vmax=255)
            # img_name = img_dir + str(i) + '.png'
            # cv2.imwrite(img_name, frame)
        cv2.waitKey(10)

    # Load a random image from the images folder
    print (os.walk(IMAGE_DIR))
    # file_names = next(os.walk(IMAGE_DIR))[2]


get_image_mask()
