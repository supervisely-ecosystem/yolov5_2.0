import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from src.models import models as yolov5_models
from src.workflow import Workflow
from ultralytics import YOLO

import supervisely as sly
from supervisely.app.widgets import (
    CustomModelsSelector,
    PretrainedModelsSelector,
    RadioTabs,
)
from supervisely.nn.prediction_dto import PredictionBBox
from supervisely.nn.artifacts.yolov5 import YOLOv5v2

try:
    from typing import Literal
except ImportError:
    # for compatibility with python 3.7
    from typing_extensions import Literal

from typing import Any, Dict, List

load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))

root_source_path = str(Path(__file__).parents[2])

api = sly.Api.from_env()
team_id = sly.env.team_id()


class YOLOv5Model(sly.nn.inference.ObjectDetection):
    def initialize_custom_gui(self):
        """Create custom GUI layout for model selection. This method is called once when the application is started."""
        self.pretrained_models_table = PretrainedModelsSelector(yolov5_models)
        sly_yolov5v2 = YOLOv5v2(team_id)
        custom_checkpoints = sly_yolov5v2.get_list()
        self.custom_models_table = CustomModelsSelector(
            team_id,
            custom_checkpoints,
            show_custom_checkpoint_path=True,
            custom_checkpoint_task_types=["object detection"],
        )

        self.model_source_tabs = RadioTabs(
            titles=["Pretrained models", "Custom models"],
            descriptions=[
                "Publicly available models",
                "Models trained by you in Supervisely",
            ],
            contents=[self.pretrained_models_table, self.custom_models_table],
        )
        return self.model_source_tabs

    def get_params_from_gui(self) -> dict:
        model_source = self.model_source_tabs.get_active_tab()
        sly.logger.debug(f"Model source: {model_source}")
        self.device = self.gui.get_device()
        if model_source == "Pretrained models":
            model_params = self.pretrained_models_table.get_selected_model_params()
        elif model_source == "Custom models":
            model_params = self.custom_models_table.get_selected_model_params()

        self.task_type = model_params.get("task_type")
        deploy_params = {"device": self.device, **model_params}
        return deploy_params

    def load_model_meta(self):
        self.class_names = list(self.model.names.values())
        obj_classes = [sly.ObjClass(name, sly.Rectangle) for name in self.class_names]
        self._model_meta = sly.ProjectMeta(
            obj_classes=sly.ObjClassCollection(obj_classes)
        )
        self._get_confidence_tag_meta()

    def load_model(
        self,
        device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"],
        model_source: Literal["Pretrained models", "Custom models"],
        task_type: Literal[
            "object detection", "instance segmentation", "pose estimation"
        ],
        checkpoint_name: str,
        checkpoint_url: str,
    ):
        """
        Load model method is used to deploy model.

        :param model_source: Specifies whether the model is pretrained or custom.
        :type model_source: Literal["Pretrained models", "Custom models"]
        :param device: The device on which the model will be deployed.
        :type device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"]
        :param task_type: The type of task the model is designed for.
        :type task_type: Literal["object detection", "instance segmentation", "pose estimation"]
        :param checkpoint_name: The name of the checkpoint from which the model is loaded.
        :type checkpoint_name: str
        :param checkpoint_url: The URL where the model can be downloaded.
        :type checkpoint_url: str
        """
        self.task_type = task_type
        local_weights_path = os.path.join(self.model_dir, checkpoint_name)
        if (
            not sly.fs.file_exists(local_weights_path)
            or model_source == "Custom models"
        ):
            self.download(
                src_path=checkpoint_url,
                dst_path=local_weights_path,
            )
        self.model = YOLO(local_weights_path)
        if device.startswith("cuda"):
            if device == "cuda":
                self.device = 0
            else:
                self.device = int(device[-1])
        else:
            self.device = "cpu"
        self.model.to(self.device)
        self.load_model_meta()

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
            results.append(
                PredictionBBox(self.class_names[cls_index], bbox, confidence)
            )
        return results


m = YOLOv5Model(
    use_gui=True,
    custom_inference_settings=os.path.join(
        root_source_path, "serve", "custom_settings.yaml"
    ),
)
workflow = Workflow(m.api)
if sly.is_production():
    m.serve()
    # -------------------------------------- Add Workflow Input -------------------------------------- #    
    workflow.add_input(m.get_params_from_gui())
    # ----------------------------------------------- - ---------------------------------------------- #
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    deploy_params = m.get_params_from_gui()
    m.load_model(**deploy_params)
    # -------------------------------------- Add Workflow Input -------------------------------------- #    
    workflow.add_input(deploy_params)
    # ----------------------------------------------- - ---------------------------------------------- #
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
