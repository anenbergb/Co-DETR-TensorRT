from typing import List, Optional
import numpy as np
import torch
from datetime import datetime

from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.visualization import Visualizer
from mmengine.registry import MODELS, VISUALIZERS

from mmdet.models.utils.misc import samplelist_boxtype2tensor


class Inferencer:
    def __init__(self, model, model_file: str):
        self.model = model

        self.cfg = Config.fromfile(model_file)

        self.data_preprocessor = MODELS.build(self.cfg.model.data_preprocessor)

        self.collate_fn = pseudo_collate
        self.visualizer = self._init_visualizer(self.cfg)

    def _init_visualizer(self, cfg: Config) -> Optional[Visualizer]:
        """Initialize visualizers.

        Args:
            cfg (ConfigType): Config containing the visualizer information.

        Returns:
            Visualizer or None: Visualizer initialized with config.
        """
        if "visualizer" not in cfg:
            return None
        timestamp = str(datetime.timestamp(datetime.now()))
        name = cfg.visualizer.get("name", timestamp)
        if Visualizer.check_instance_created(name):
            name = f"{name}-{timestamp}"
        cfg.visualizer.name = name
        visualizer = VISUALIZERS.build(cfg.visualizer)

        # check whether this is works
        visualizer.dataset_meta = self.model.dataset_meta
        return visualizer

    def add_pred_to_datasample(self, data_samples, results_list):
        """Add predictions to `DetDataSample`.

        Args:
            data_samples (list[:obj:`DetDataSample`], optional): A batch of
                data samples that contain annotations and predictions.
            results_list (list[:obj:`InstanceData`]): Detection results of
                each image.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        for data_sample, pred_instances in zip(data_samples, results_list):
            data_sample.pred_instances = pred_instances
        samplelist_boxtype2tensor(data_samples)
        return data_samples

    def __call__(
        self,
        images: List[np.ndarray],
        return_vis: bool = False,
        show: bool = False,
        wait_time: int = 0,
        no_save_vis: bool = False,
        draw_pred: bool = True,
        pred_score_thr: float = 0.3,
        return_datasamples: bool = False,
        print_result: bool = False,
        no_save_pred: bool = True,
        out_dir: str = "",
    ):
        results_dict = {"predictions": [], "visualization": []}

        inputs = [self.collate_fn([image]) for image in images]  # returns list of lists
        for ori_imgs, data in inputs:
            # ori_imgs are the numpy images in the batch, which in this case is just one image
            # data is dict containing keys "inputs" and "data_samples"
            # data["inputs"] is a list of torch tensors of shape (3, H, W)
            # data["data_samples"] is a list of DetDataSample objects
            with torch.no_grad():
                data_processed = self.data_preprocessor(data, False)
                batch_inputs = data_processed["inputs"]
                batch_data_samples = data_processed["data_samples"]
                results_list = self.model(batch_inputs, batch_data_samples)

            preds = self.add_pred_to_datasample(batch_data_samples, results_list)

            visualization = self.visualize(
                ori_imgs,
                preds,
                return_vis=return_vis,
                show=show,
                wait_time=wait_time,
                draw_pred=draw_pred,
                pred_score_thr=pred_score_thr,
                no_save_vis=no_save_vis,
                img_out_dir=out_dir,
            )
            results = self.postprocess(
                preds,
                visualization,
                return_datasamples=return_datasamples,
                print_result=print_result,
                no_save_pred=no_save_pred,
                pred_out_dir=out_dir,
            )
            results_dict["predictions"].extend(results["predictions"])
            if results["visualization"] is not None:
                results_dict["visualization"].extend(results["visualization"])
        return results_dict
