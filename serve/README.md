<div align="center" markdown>
<img src="https://github.com/supervisely-ecosystem/yolov5_2.0/assets/115161827/b31718cf-560a-4924-b048-dbfc57748898"/>  

# Serve YOLOv5 2.0

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#example-apply-yolov5-20-to-image-in-labeling-tool">Example: apply YOLOv5 2.0 to image in labeling tool</a> •
  <a href="#Related-apps">Related Apps</a> •
  <a href="#Acknowledgment">Acknowledgment</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervisely.com/apps/supervisely-ecosystem/yolov5_2.0/serve)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervisely.com/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/yolov5_2.0)
[![views](https://app.supervisely.com/img/badges/views/supervisely-ecosystem/yolov5_2.0/serve.png)](https://supervisely.com)
[![runs](https://app.supervisely.com/img/badges/runs/supervisely-ecosystem/yolov5_2.0/serve.png)](https://supervisely.com)

</div>

# Overview

This app allows to use updated YOLOv5 checkpoints, which show better performance in comparison to thier previous versions. App supports both models pretrained on COCO and models trained on custom datasets.

# How To Run

## Pretrained models

**Step 1.** Select pretrained model architecture and press the **Serve** button

![screenshot-dev-supervise-ly-apps-9988-sessions-46331-1698920936898](https://github.com/supervisely-ecosystem/yolov5_2.0/assets/115161827/4e9333dd-8be6-45c8-a7e1-f5bc60dfed97)

**Step 2.** Wait for the model to deploy

![screenshot-dev-supervise-ly-apps-9988-sessions-46331-1698920945183](https://github.com/supervisely-ecosystem/yolov5_2.0/assets/115161827/c050c7eb-0d16-44c0-94cd-05303af0b48e)

## Custom models

Copy model file path from Team Files and select task type:

todo

# Example: apply YOLOv5 to image in labeling tool

Run **NN Image Labeling** app, connect to YOLOv5 app session, and click on "Apply model to image", or if you want to apply model only to the region within the bounding box, select the bbox and click on "Apply model to ROI":

https://github.com/supervisely-ecosystem/yolov5_2.0/assets/115161827/e81d353d-820f-46ec-b5bb-5470d4f19cfe

If you want to change model specific inference settings while working with the model in image labeling interface, go to **inference** tab in the settings section of **Apps** window, and change the parameters:
<p align="center">
<img src="https://github.com/supervisely-ecosystem/yolov5_2.0/assets/115161827/e037223a-f6d3-4725-be70-a7ef743e13ff" width=50% />
</p>

# Related apps

- [NN Image Labeling](../../../../supervisely-ecosystem/supervisely-ecosystem%252Fnn-image-labeling%252Fannotation-tool) - integrate any deployed NN to Supervisely Image Labeling UI. Configure inference settings and model output classes. Press `Apply` button (or use hotkey) and detections with their confidences will immediately appear on the image.   
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/nn-image-labeling/annotation-tool" src="https://i.imgur.com/hYEucNt.png" height="70px" margin-bottom="20px"/>

- [Apply NN to Videos Project](../../../../supervisely-ecosystem/apply-nn-to-videos-project) - app allows to label your videos using served Supervisely models.  
  <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/apply-nn-to-videos-project" src="https://imgur.com/LDo8K1A.png" height="70px" margin-bottom="20px" />

- [Train YOLOv5 2.0](https://ecosystem.supervisely.com/apps/supervisely-ecosystem/yolov5_2.0/train) - app allows to create custom YOLOv5 weights through training process.
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/yolov5_2.0/train" src="https://github.com/supervisely-ecosystem/yolov5_2.0/assets/115161827/b9352f75-f3c4-485c-a763-91a7f8401f09" height="70px" margin-bottom="20px"/>

    
# Acknowledgment

This app is based on the great work `YOLOv5` ([github](https://github.com/ultralytics/ultralytics)). ![GitHub Org's stars](https://img.shields.io/github/stars/ultralytics/ultralytics?style=social)




