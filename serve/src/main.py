import os
from pathlib import Path

import supervisely as sly
import torch
from dotenv import load_dotenv
from supervisely.annotation.obj_class import ObjClass
from supervisely.app.widgets import Field, RadioGroup
from supervisely.imaging.color import get_predefined_colors
from supervisely.nn.prediction_dto import (PredictionBBox, PredictionKeypoints,
                                           PredictionMask)
from supervisely.project.project_meta import ProjectMeta
from ultralytics import YOLO

# for local inference
# from keypoints_template import human_template, dict_to_template

try:
    from typing import Literal
except ImportError:
    # for compatibility with python 3.7
    from typing_extensions import Literal

from typing import Any, Dict, List, Union

import numpy as np

#load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))
root_source_path = str(Path(__file__).parents[2])
det_models_data_path = os.path.join(root_source_path, "models", "det_models_data.json")
det_models_data = sly.json.load_json_file(det_models_data_path)


class YOLOv5Model(sly.nn.inference.ObjectDetection):
    def get_models(self):
        return det_models_data

    def load_on_device(
        self,
        model_dir,
        device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"] = "cpu",
    ):
        model_source = self.gui.get_model_source()
        if model_source == "Pretrained models":
            selected_model = self.gui.get_checkpoint_info()["Model"]
            model_filename = selected_model.lower() + ".pt"
            weights_url = (
                f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{model_filename}"
            )
            weights_dst_path = os.path.join(model_dir, model_filename)
            if not sly.fs.file_exists(weights_dst_path):
                self.download(
                    src_path=weights_url,
                    dst_path=weights_dst_path,
                )
        elif model_source == "Custom models":
            custom_link = self.gui.get_custom_link()
            weights_file_name = os.path.basename(custom_link)
            weights_dst_path = os.path.join(model_dir, weights_file_name)
            if not sly.fs.file_exists(weights_dst_path):
                self.download(
                    src_path=custom_link,
                    dst_path=weights_dst_path,
                )
        self.model = YOLO(weights_dst_path)
        self.class_names = list(self.model.names.values())
        self.task_type = "object detection"
        if device.startswith("cuda"):
            if device == "cuda":
                self.device = 0
            else:
                self.device = int(device[-1])
        else:
            self.device = "cpu"
        self.model.to(self.device)
        print(f"✅ Model has been successfully loaded on {device.upper()} device")

    def get_info(self):
        info = super().get_info()
        info["task type"] = self.task_type
        info["videos_support"] = True
        info["async_video_inference_support"] = True
        info["tracking_on_videos_support"] = True
        if self.task_type == "pose estimation":
            info["detector_included"] = True
        return info

    def get_classes(self) -> List[str]:
        return self.class_names

    @property
    def model_meta(self) -> ProjectMeta:
        if self._model_meta is None:
            colors = get_predefined_colors(len(self.get_classes()))
            classes = []
            for name, rgb in zip(self.get_classes(), colors):
                classes.append(ObjClass(name, sly.Rectangle, rgb))
            self._model_meta = ProjectMeta(classes)
            self._get_confidence_tag_meta()
        return self._model_meta

    def _create_label(self, dto: PredictionBBox):
        obj_class = self.model_meta.get_obj_class(dto.class_name)
        if obj_class is None:
            raise KeyError(
                f"Class {dto.class_name} not found in model classes {self.get_classes()}"
            )
        geometry = sly.Rectangle(*dto.bbox_tlbr)
        tags = []
        if dto.score is not None:
            tags.append(sly.Tag(self._get_confidence_tag_meta(), dto.score))
        label = sly.Label(geometry, obj_class, tags)
        return label

    def predict(
        self, image_path: str, settings: Dict[str, Any]
    ) -> List[PredictionBBox]:
        input_image = sly.image.read(image_path)
        # RGB to BGR
        input_image = input_image[:, :, ::-1]
        input_height, input_width = input_image.shape[:2]
        predictions = self.model(
            source=input_image,
            conf=settings["conf"],
            iou=settings["iou"],
            half=settings["half"],
            device=self.device,
            max_det=settings["max_det"],
            agnostic_nms=settings["agnostic_nms"],
        )
        results = []
        boxes_data = predictions[0].boxes.data
        for box in boxes_data:
            left, top, right, bottom, confidence, cls_index = (
                int(box[0]),
                int(box[1]),
                int(box[2]),
                int(box[3]),
                float(box[4]),
                int(box[5]),
            )
            bbox = [top, left, bottom, right]
            results.append(PredictionBBox(self.class_names[cls_index], bbox, confidence))
        return results


m = YOLOv5Model(
    use_gui=True,
    custom_inference_settings=os.path.join(root_source_path, "serve", "custom_settings.yaml"),
)

if sly.is_production():
    m.serve()
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    m.load_on_device(m.model_dir, device)
    image_path = "./demo_data/image_01.jpg"
    settings = {
        "conf": 0.25,
        "iou": 0.7,
        "half": False,
        "max_det": 300,
        "agnostic_nms": False,
    }
    results = m.predict(image_path, settings=settings)
    vis_path = "./demo_data/image_01_prediction.jpg"
    m.visualize(results, image_path, vis_path, thickness=5)
    print(f"predictions and visualization have been saved: {vis_path}")
