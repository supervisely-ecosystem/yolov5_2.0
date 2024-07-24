import supervisely as sly
import os


def check_compatibility(func):
    def wrapper(self, *args, **kwargs):
        if self.is_compatible is None:
            try:
                self.is_compatible = self.check_instance_ver_compatibility()
            except Exception as e:
                sly.logger.error(
                    "Can not check compatibility with Supervisely instance. "
                    f"Workflow features will be disabled. Error: {repr(e)}"
                )
                self.is_compatible = False
        if not self.is_compatible:
            return
        return func(self, *args, **kwargs)

    return wrapper


class Workflow:
    def __init__(self, api: sly.Api, min_instance_version: str = None):
        self.is_compatible = None
        self.api = api
        self.is_model_weight_added = False
        self._min_instance_version = (
            "6.9.31" if min_instance_version is None else min_instance_version
        )
    
    def check_instance_ver_compatibility(self):
        if not self.api.is_version_supported(self._min_instance_version):
            sly.logger.info(
                f"Supervisely instance version {self.api.instance_version} does not support workflow features."
            )
            if not sly.is_community():
                sly.logger.info(
                    f"To use them, please update your instance to version {self._min_instance_version} or higher."
                )
            return False
        return True

    @check_compatibility
    def add_input(self, deploy_params: dict):
        try:
            if self.is_model_weight_added:
                return
            model_source = deploy_params.get("model_source")
            sly.logger.debug(f"Deploy Params - {deploy_params}")
            if model_source == "Custom models":
                checkpoint_url = deploy_params.get("checkpoint_url")
                meta = {"customNodeSettings": {"title": "<h4>Serve Custom Model</h4>"}}
                sly.logger.debug(f"Workflow Input: Checkpoint URL - {checkpoint_url}")
                if self.api.file.exists(sly.env.team_id(), checkpoint_url):
                    customization_file = os.path.join(os.path.dirname(checkpoint_url), "workflow.json")
                    if self.api.file.exists(sly.env.team_id(), customization_file):
                        meta["customRelationSettings"] = self.api.file.get_json_file_content(sly.env.team_id(), customization_file)
                    response = self.api.app.workflow.add_input_file(checkpoint_url, model_weight=True, meta=meta)
                    if response.get("id", None) is not None:
                        self.is_model_weight_added = True
                else:
                    sly.logger.debug(f"Checkpoint {checkpoint_url} not found in Team Files. Cannot set workflow input")
        except Exception as e:
            sly.logger.debug(f"Failed to add input to the workflow: {repr(e)}")
            
    @check_compatibility
    def add_output(self):
        raise NotImplementedError("add_output is not implemented in this workflow")
