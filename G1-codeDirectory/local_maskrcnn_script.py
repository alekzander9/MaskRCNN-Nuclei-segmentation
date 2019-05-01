import os
import sys
import random
import math
import re
import time
import numpy as np

from nuclei import NucleiConfig, NucleiDataset
from nucleitools import NucleiTools
import utils
import model as modellib
from keras.callbacks import EarlyStopping, ModelCheckpoint
from model import log


#Training or Evaluating?
command = "train"

# Which weights to start with?
init_with = "coco"  # imagenet, coco, last, or specific

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "pretrained_mask_rcnn_coco.h5")

#Define a particular set of weights
SPECIFIC_MODEL_PATH = os.path.join(ROOT_DIR,"mask_rcnn_nuclei_xxx.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
if command == "train":
    # Training dataset
    dataset_train = NucleiDataset()
    dataset_train.load_nuclei('../data/stage1_train/', 'train')
    dataset_train.prepare()

    # Validation dataset
    dataset_val = NucleiDataset()
    dataset_val.load_nuclei('../data/stage1_train/', 'val')
    dataset_val.prepare()

else:
    # Validation dataset
    dataset_test = NucleiDataset()
    dataset_test.load_nuclei('../data/stage1_test/', 'test',improved=True)
    dataset_test.prepare()

# Configurations
if command == "train":
    config = NucleiConfig()
    checkpointer = ModelCheckpoint('model-mask-2018-1.h5', verbose=1, save_best_only=True)
    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config,
                            model_dir=MODEL_DIR)
else:
    #Define an inference config. This should be the same as that used to train with
    #With the exception of using 1 Image per GPU
    class InferenceConfig(NucleiConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        # Number of classes (including background)
        NUM_CLASSES = 1 + 1  # background + nuclei

        # Use small images for faster training. Set the limits of the small side
        # the large side, and that determines the image shape.
        IMAGE_MIN_DIM = 512
        IMAGE_MAX_DIM = 512

        # Use smaller anchors because our image and objects are small
        RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

        # Reduce training ROIs per image because the images are small and have
        # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
        TRAIN_ROIS_PER_IMAGE = 500

        # Use a small epoch since the data is simple
        STEPS_PER_EPOCH = 100

        # use small validation steps since the epoch is small
        VALIDATION_STEPS = 20

        #USE_MINI_MASK = False

        RPN_ANCHOR_STRIDE = 1

        MAX_GT_INSTANCES = 300

        DETECTION_MAX_INSTANCES = 300

        BACKBONE = "resnet101"

        DETECTION_NMS_THRESHOLD = 0.6

    config = InferenceConfig()
    # Create model in inference mode
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                            config=config)
config.display()                          

#Choose which weights to load
if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    last_found = model.find_last()[1]
    if(last_found==None):
        print("Could not find last weights. Exiting")
        exit()
    print("Checkpoint Weights: " + last_found)
    model.load_weights(model.find_last()[1], by_name=True)
   
elif init_with == "specific":
    model.load_weights(SPECIFIC_MODEL_PATH, by_name=True)


#Select whether we are training or inferring
if command == "train":
    # Training - Stage 1
    print("Training network heads")
    #If resuming training, set which epoch to start from.
    #model.epoch = 17
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                layers='heads')
    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 4 and up")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=120,
                layers='4+')
    # Training - Stage 3
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=160,
                layers='all')

elif command == "evaluate":
    import pandas as pd
    from skimage.transform import resize
    from skimage.morphology import label
    def rle_encoding(x):
            dots = np.where(x.T.flatten() == 1)[0]
            run_lengths = []
            prev = -2
            for b in dots:
                if (b>prev+1): run_lengths.extend((b + 1, 0))
                run_lengths[-1] += 1
                prev = b
            return run_lengths

    def prob_to_rles(x, cutoff=0.5):
        lab_img = label(x > cutoff)
        for i in range(1, lab_img.max() + 1):
            yield rle_encoding(lab_img == i)

    #Creat list containing results. This is a list of lists of dictionaries
    results=[]
    for image_id in dataset_test.image_ids:
        image = dataset_test.load_image(image_id)
        info = dataset_test.image_info[image_id]
        print("Processing image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
                                       dataset_test.image_reference(image_id)))
        # Run object detection
        results.append(model.detect([image], verbose=1))
    print("Finished Evaluation")

    #Create two lists. One with the ids of the original images. The other with a single mask for each
    test_ids = [] 
    mask_list = []
    for image in range(len(results)):
        merged_mask = np.zeros(results[image][0]['masks'].shape[0:2])
        for mask in range(results[image][0]['masks'].shape[-1]):
            merged_mask = np.ma.mask_or(merged_mask.astype(np.uint8),(results[image][0]['masks'][:,:,mask]>0.5).astype(np.uint8))
        print("Merged "+str(results[image][0]['masks'].shape[-1])+" nuclei masks for image "+str(image))
        mask_list.append(merged_mask)
        info = dataset_test.image_info[image]
        test_ids.append(info["id"])
    new_test_ids = []
    rles = []
    for n, id_ in enumerate(test_ids):
        rle = list(prob_to_rles(mask_list[n]))
        rles.extend(rle)
        new_test_ids.extend([id_] * len(rle))

    output_file= "submission.csv"
    # Create submission DataFrames
    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    sub.to_csv(output_file, index=False)
else:
    print ("Command "+ command + " Unknown")
    exit()

