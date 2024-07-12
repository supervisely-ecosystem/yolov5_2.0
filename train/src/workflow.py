# Description: This file contains versioning features and the Workflow class that is used to add input and output to the workflow.

import supervisely as sly
import os


def check_compatibility(func):
    def wrapper(self, *args, **kwargs):
        if self.is_compatible is None:
            self.is_compatible = self.check_instance_ver_compatibility()
        if not self.is_compatible:
            return
        return func(self, *args, **kwargs)

    return wrapper


class Workflow:
    def __init__(self, api: sly.Api, min_instance_version: str = None):
        self.is_compatible = None
        self.api = api
        self._min_instance_version = (
            "6.9.31" if min_instance_version is None else min_instance_version
        )
    
    def check_instance_ver_compatibility(self):
        if not self.api.is_version_supported(self._min_instance_version):
            sly.logger.info(
                f"Supervisely instance version {self.api.instance_version} does not support workflow and versioning features."
            )
            if not sly.is_community():
                sly.logger.info(
                    f"To use them, please update your instance to version {self._min_instance_version} or higher."
                )
            return False
        return True
    
    @check_compatibility
    def add_input(self, project_info: sly.ProjectInfo, weight_file: str = None):
        try:
            project_version_id = self.api.project.version.create(
                project_info, "Train YOLO v5 2.0", f"This backup was created automatically by Supervisely before the Train YOLO task with ID: {self.api.task_id}"
            )
        except Exception as e:
            sly.logger.error(f"Failed to create a project version: {e}")
            project_version_id = None
            
        try:
            if project_version_id is None:
                project_version_id = project_info.version.get("id", None) if project_info.version else None
            self.api.app.workflow.add_input_project(project_info.id, version_id=project_version_id)
            file_info = False
            if weight_file:
                self.api.app.workflow.add_input_file(weight_file, model_weight=True)
            sly.logger.debug(f"Workflow Input: Project ID - {project_info.id}, Project Version ID - {project_version_id}, Input File - {True if file_info else False}")
        except Exception as e:
            sly.logger.error(f"Failed to add input to the workflow: {e}")

    @check_compatibility
    def add_output(self, team_files_dir: str, app_url: str, weights_type: str = ''):
        try:
            weights_team_files_dir = os.path.join(team_files_dir, "weights")
            file_infos = self.api.file.list(sly.env.team_id(), weights_team_files_dir, return_type="fileinfo")
            best_file_info = None
            for file_info in file_infos:
                if "best" in file_info.name:
                    best_file_info = file_info
                    break
            if best_file_info:
                if weights_type == "Custom models":
                    model_name = "Custom Model"
                else:
                    model_name = "YOLOv5 2.0"
                
                meta = {
                    "customNodeSettings": {
                    "title": f"<h4>Train {model_name}</h4>",
                    "mainLink": {
                        "url": app_url,
                        "title": "Show Results"
                    }
                },
                "customRelationSettings": {
                    "icon": {
                        "icon": "zmdi-folder",
                        "color": "#FFA500",
                        "backgroundColor": "#FFE8BE"
                    },
                    "title": "<h4>Checkpoints</h4>",
                    "mainLink": {"url": f"/files/{best_file_info.id}/true", "title": "Open Folder"}
                    }
                }
                sly.logger.debug(f"Workflow Output: Best file path - {best_file_info.path}, Best file name - {best_file_info.name}, App URL - {app_url}")
                sly.logger.debug(f"Workflow Output: meta \n    {meta}")
                self.api.app.workflow.add_output_file(best_file_info, model_weight=True, meta=meta)
            else:
                sly.logger.debug(f"File with the best weighs not found in Team Files. Cannot set workflow output.")
        except Exception as e:
            sly.logger.error(f"Failed to add output to the workflow: {e}")
