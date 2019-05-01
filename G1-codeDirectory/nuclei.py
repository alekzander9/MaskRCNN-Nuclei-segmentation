"""
Mask R-CNN
Configurations and data loading code for the Nuclei dataset.
"""

import math
import random
import numpy as np

from config import Config
import utils
import os

from nucleitools import NucleiTools


import model as modellib

class NucleiConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "nuclei"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    #If one 12GB can fit two 1024x1024 images, then it should be able to fit 32 256x256
    #Set to 16 to match original paper
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + nuclei

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 50

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 20

    RPN_ANCHOR_STRIDE = 1

    RPN_ANCHOR_STRIDE = 1

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 300

    # Max number of final detections
    # There may be more than 100 nuclei in one image
    DETECTION_MAX_INSTANCES = 300

    # Non-max suppression threshold to filter RPN proposals.
    # You can reduce this during training to generate more propsals.
    # 0.5 matches the original paper
    RPN_NMS_THRESHOLD = 0.6

    # How many anchors per image to use for RPN training
    # Matched this such that it will choose 1 ROI at each point with the best shape
    #RPN_TRAIN_ANCHORS_PER_IMAGE = 128

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 500

    ROI_POSITIVE_RATIO = 0.4

    BACKBONE = "resnet50"

    DETECTION_MAX_INSTANCES = 300

    DETECTION_NMS_THRESHOLD = 0.2

class NucleiDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """
    #data_type either 'val' or 'train'
    def load_nuclei(self, dataset_dir, data_type, improved=True):
        """Generate the requested number of synthetic images.
        """
        # Add classes
        self.add_class("nuclei", 1, "nucleus")

        # Add images
        # Generate random specifications of images (i.e. color and
        # list of shapes sizes and locations). This is more compact than
        # actual images. Images are generated on the fly in load_image().
        nuclei = NucleiTools(dataset_dir,improved=improved)
        nuclei.load_images(data_type)
        if data_type != "test":
            nuclei.load_masks(data_type)

        for _id, nucleus in nuclei.imgs.items():
            self.add_image(
                source="nuclei",
                path= os.path.join(dataset_dir, _id, "images", nucleus['filename']),
                image_id=_id,
                width = nucleus['width'],
                height = nucleus['height'],
                masks = nucleus['masks']
            )

    # TODO Currently unimplemented because its utility has yet to be understood for our dataset.
    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "nuclei":
            # TODO return the image id path or something of the sort. This is used for debugging (described in parent class).
            return ""
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        image_info = self.image_info[image_id]

        if image_info["source"] != "nuclei":
            return super(self.__class__).load_mask(image_id)
        instance_masks = image_info['masks']

        mask = np.stack(instance_masks, axis=2)
        class_ids = np.ones(len(instance_masks), dtype=np.int32)
        return mask, class_ids
