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
from supervisely.app.content import get_data_dir

import supervisely.nn.inference.gui as GUI
from supervisely.nn.inference.inference import (
    LOAD_ON_DEVICE_DECORATOR, 
    LOAD_MODEL_DECORATOR, 
    add_callback, 
    Inference, 
    InferenceImageCache, 
    ThreadPoolExecutor, 
    env,
)
from supervisely.task.progress import Progress

try:
    from typing import Literal, Union, Optional
except ImportError:
    # for compatibility with python 3.7
    from typing_extensions import Literal, Union, Optional

from typing import Any, Dict, List

load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))

root_source_path = str(Path(__file__).parents[2])

api = sly.Api.from_env()
team_id = sly.env.team_id()


class YOLOv5Model(sly.nn.inference.ObjectDetection):    
    def __init__(
        self,
        model_dir: Optional[str] = None,
        custom_inference_settings: Optional[
            Union[Dict[str, Any], str]
        ] = None,  # dict with settings or path to .yml file
        sliding_window_mode: Optional[Literal["basic", "advanced", "none"]] = "basic",
        use_gui: Optional[bool] = False,
        multithread_inference: Optional[bool] = True,
    ):
        if model_dir is None:
            model_dir = os.path.join(get_data_dir(), "models")
            sly.fs.mkdir(model_dir)
        self._model_dir = model_dir
        self._model_served = False
        self._model_meta = None
        self._confidence = "confidence"
        self._app: sly.Application = None
        self._api: sly.Api = None
        self._task_id = None
        self._sliding_window_mode = sliding_window_mode
        self._autostart_delay_time = 5 * 60  # 5 min
        self._tracker = None
        if custom_inference_settings is None:
            custom_inference_settings = {}
        if isinstance(custom_inference_settings, str):
            if sly.fs.file_exists(custom_inference_settings):
                with open(custom_inference_settings, "r") as f:
                    custom_inference_settings = f.read()
            else:
                raise FileNotFoundError(f"{custom_inference_settings} file not found.")
        self._custom_inference_settings = custom_inference_settings

        self._use_gui = use_gui
        self._gui = None

        self.load_on_device = LOAD_ON_DEVICE_DECORATOR(self.load_on_device)
        self.load_on_device = add_callback(self.load_on_device, self._set_served_callback)

        self.load_model = LOAD_MODEL_DECORATOR(self.load_model)
        self.load_model = add_callback(self.load_model, self._set_served_callback)

        if use_gui:
            initialize_custom_gui_method = getattr(self, "initialize_custom_gui", None)
            original_initialize_custom_gui_method = getattr(
                Inference, "initialize_custom_gui", None
            )
            if initialize_custom_gui_method.__func__ is not original_initialize_custom_gui_method:
                self._gui = GUI.ServingGUI()
                self._user_layout = self.initialize_custom_gui()
            else:
                self.initialize_gui()

            def on_serve_callback(gui: Union[GUI.InferenceGUI, GUI.ServingGUI]):
                Progress("Deploying model ...", 1)

                if isinstance(self.gui, GUI.ServingGUI):
                    deploy_params = self.get_params_from_gui()
                    self.load_model(**deploy_params)
                    # -------------------------------------- Add Workflow Input -------------------------------------- #    
                    workflow.add_input(deploy_params)
                    # ----------------------------------------------- - ---------------------------------------------- #
                    self.update_gui(self._model_served)
                else:  # GUI.InferenceGUI
                    device = gui.get_device()
                    self.load_on_device(self._model_dir, device)
                gui.show_deployed_model_info(self)

            def on_change_model_callback(gui: Union[GUI.InferenceGUI, GUI.ServingGUI]):
                self.shutdown_model()
                if isinstance(self.gui, GUI.ServingGUI):
                    self._api_request_model_layout.unlock()
                    self._api_request_model_layout.hide()
                    self.update_gui(self._model_served)
                    self._user_layout_card.show()

            self.gui.on_change_model_callbacks.append(on_change_model_callback)
            self.gui.on_serve_callbacks.append(on_serve_callback)

        self._inference_requests = {}
        max_workers = 1 if not multithread_inference else None
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self.predict = self._check_serve_before_call(self.predict)
        self.predict_raw = self._check_serve_before_call(self.predict_raw)
        self.get_info = self._check_serve_before_call(self.get_info)

        self.cache = InferenceImageCache(
            maxsize=env.smart_cache_size(),
            ttl=env.smart_cache_ttl(),
            is_persistent=True,
            base_folder=env.smart_cache_container_dir(),
        )

    # def __init__(self, *args, **kwargs):
    #     super(YOLOv5Model, self).__init__(*args, **kwargs)
    #     if self._use_gui:
    #         def on_serve_callback(gui: Union[GUI.InferenceGUI, GUI.ServingGUI]):
    #             Progress("Deploying model ...", 1)

    #             if isinstance(self.gui, GUI.ServingGUI):
    #                 deploy_params = self.get_params_from_gui()
    #                 # -------------------------------------- Add Workflow Input -------------------------------------- #    
    #                 workflow.add_input(deploy_params)
    #                 # ----------------------------------------------- - ---------------------------------------------- #
    #                 self.load_model(**deploy_params)
    #                 self.update_gui(self._model_served)
    #             else:  # GUI.InferenceGUI
    #                 device = gui.get_device()
    #                 self.load_on_device(self._model_dir, device)
    #             gui.show_deployed_model_info(self)
    #         self.gui.on_serve_callbacks.append(on_serve_callback)

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
