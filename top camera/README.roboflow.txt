
Drone detection - v2 2022-04-01 12:22pm
==============================

This dataset was exported via roboflow.ai on April 1, 2022 at 9:26 AM GMT

It includes 47 images.
Dji-tello-edu-drone are annotated in YOLO v5 PyTorch format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* 50% probability of vertical flip
* Equal probability of one of the following 90-degree rotations: none, clockwise, counter-clockwise, upside-down
* Random rotation of between -45 and +45 degrees
* Random brigthness adjustment of between -41 and +41 percent


