import supervisely as sly
from supervisely.app.widgets import RadioGroup, Field
from supervisely.project.project_meta import ProjectMeta
from supervisely.annotation.obj_class import ObjClass
from supervisely.imaging.color import get_predefined_colors
from supervisely.nn.prediction_dto import PredictionMask, PredictionBBox, PredictionKeypoints
from ultralytics import YOLO
import torch
import os
from dotenv import load_dotenv
from pathlib import Path

from src.keypoints_template import human_template, dict_to_template

# for local inference
# from keypoints_template import human_template, dict_to_template

try:
    from typing import Literal
except ImportError:
    # for compatibility with python 3.7
    from typing_extensions import Literal
from typing import List, Any, Dict, Union
import numpy as np


load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))
root_source_path = str(Path(__file__).parents[2])
det_models_data_path = os.path.join(root_source_path, "models", "det_models_data.json")
seg_models_data_path = os.path.join(root_source_path, "models", "seg_models_data.json")
pose_models_data_path = os.path.join(root_source_path, "models", "pose_models_data.json")
det_models_data = sly.json.load_json_file(det_models_data_path)
seg_models_data = sly.json.load_json_file(seg_models_data_path)
pose_models_data = sly.json.load_json_file(pose_models_data_path)


class YOLOv5Model(sly.nn.inference.ObjectDetection):
    def add_content_to_pretrained_tab(self, gui):
        task_type_items = [
            RadioGroup.Item(value="object detection"),
        ]
        self.task_type_select = RadioGroup(items=task_type_items, direction="vertical")
        task_type_select_f = Field(
            content=self.task_type_select,
            title="Task type",
        )

        @self.task_type_select.value_changed
        def change_table(task_type):
            models_table_columns = [key for key in det_models_data[0].keys()]
            models_table_subtitles = [None] * len(models_table_columns)
            models_table_rows = []
            for element in det_models_data:
                models_table_rows.append(list(element.values()))
            gui._models_table.set_data(
                columns=models_table_columns,
                rows=models_table_rows,
                subtitles=models_table_subtitles,
            )
            gui._models = det_models_data

        return task_type_select_f

    def add_content_to_custom_tab(self, gui):
        task_type_items = [
            RadioGroup.Item(value="object detection"),
        ]
        self.custom_task_type_select = RadioGroup(items=task_type_items, direction="vertical")
        custom_task_type_select_f = Field(
            content=self.custom_task_type_select,
            title="Task type",
        )
        return custom_task_type_select_f

    def get_models(self):
        return det_models_data

    def load_on_device(
        self,
        model_dir,
        device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"] = "cpu",
    ):
        model_source = self.gui.get_model_source()
        if model_source == "Pretrained models":
            self.task_type = self.task_type_select.get_value()
            selected_model = self.gui.get_checkpoint_info()["Model"]
            if selected_model.endswith("det"):
                selected_model = selected_model[:-4]
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
            self.task_type = self.custom_task_type_select.get_value()
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
        info["tracking_on_videos_support"] = False
        if self.task_type == "pose estimation":
            info["detector_included"] = True
        return info

    def get_classes(self) -> List[str]:
        return self.class_names

    @property
    def model_meta(self) -> ProjectMeta:
        if self._model_meta is None:
            if self.task_type in ["object detection"]:
                colors = get_predefined_colors(len(self.get_classes()))
                classes = []
                for name, rgb in zip(self.get_classes(), colors):
                    if self.task_type == "object detection":
                        classes.append(ObjClass(name, sly.Rectangle, rgb))
                    else:
                        classes.append(ObjClass(name, sly.Bitmap, rgb))
                self._model_meta = ProjectMeta(classes)
                self._get_confidence_tag_meta()
        return self._model_meta

    def _create_label(self, dto: PredictionBBox):
        if self.task_type == "object detection":
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
