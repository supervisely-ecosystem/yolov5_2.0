<div align="center" markdown>

<img src="https://github.com/supervisely-ecosystem/yolov5_2.0/assets/115161827/eea6188c-ea1e-474d-b588-9ac5f6bc2685"/>

# Train YOLOv5 2.0

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#How-To-Use-Your-Trained-Model-Outside-Supervisely">How To Use Your Trained Model Outside Supervisely</a> •
  <a href="#Related-apps">Related apps</a> •
  <a href="#Screenshot">Screenshot</a> •
  <a href="#Acknowledgment">Acknowledgment</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/yolov5_2.0/train)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/yolov5_2.0)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/yolov5_2.0/train.png)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/yolov5_2.0/train.png)](https://supervise.ly)

</div>

# Overview

This app allows to train YOLOv5 model on selected dataset using updated YOLOv5 checkpoints, which show better performance in comparison to thier previous versions. You can define model checkpoint, data split method, training hyperparameters, data augmentation and many other features related to model training. App supports both models pretrained on COCO and models trained on custom datasets.

# How To Run

Select images project, select GPU device in "Agent" field, click on RUN button:

https://user-images.githubusercontent.com/91027877/249008934-293b3176-d5f3-4edb-9816-15bffd3bb869.mp4

# How To Use Your Trained Model Outside Supervisely

You can use your trained models outside Supervisely platform without any dependencies on Supervisely SDK. See this [Jupyter Notebook](https://github.com/supervisely-ecosystem/yolov5_2.0/blob/master/outside_supervisely/inference_outside_supervisely.ipynb) for details.

# Related apps

- [Serve YOLOv5 2.0](https://ecosystem.supervise.ly/apps/yolov5_2.0/serve) - app allows to deploy YOLOv8 model as REST API service.   
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/yolov5_2.0/serve" src="https://github.com/supervisely-ecosystem/yolov5_2.0/assets/115161827/90815cf8-7071-4215-9fe0-d9f1c4050e57" height="70px" margin-bottom="20px"/>
    
# Screenshot

![screencapture-dev-supervise-ly-apps-9995-sessions-46469-2023-11-03-12_53_13](https://github.com/supervisely-ecosystem/yolov5_2.0/assets/115161827/22befebd-b36d-45a0-8b03-009c5e13587c)


# Acknowledgment

This app is based on the great work `YOLOv5` ([github](https://github.com/ultralytics/ultralytics)). ![GitHub Org's stars](https://img.shields.io/github/stars/ultralytics/ultralytics?style=social)
