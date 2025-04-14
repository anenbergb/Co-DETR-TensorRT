import os.path as osp
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import mmcv
import mmengine
import numpy as np
import torch
from mmcv.transforms import LoadImageFromFile
from mmdet.models.data_preprocessors import DetDataPreprocessor
from mmdet.models.utils.misc import samplelist_boxtype2tensor
from mmdet.structures import DetDataSample, SampleList
from mmdet.structures.mask import encode_mask_results, mask2bbox
from mmdet.utils import InstanceList
from mmengine.config import Config
from mmengine.dataset import Compose, pseudo_collate
from mmengine.registry import VISUALIZERS
from mmengine.structures import InstanceData
from mmengine.visualization import Visualizer
from torchvision.ops import batched_nms

# SampleList is a list of DetDataSample
# InstanceList is a list of InstanceData


class Inferencer:
    """A class for running inference on object detection models.

    This class handles preprocessing, inference, postprocessing, and visualization
    of predictions for object detection models. It supports custom pipelines,
    score thresholds, IoU thresholds, and visualization options.

    Args:
        model (nn.Module): The object detection model to use for inference.
        model_file (str): Path to the model configuration file.
        dataset_meta (dict): Metadata for the dataset (e.g., class names).
        score_threshold (float, optional): Minimum score threshold for predictions.
            If not provided, the value from the model's test configuration is used.
        iou_threshold (float, optional): IoU threshold for non-maximum suppression (NMS).
            If not provided, the value from the model's test configuration is used.
    """

    def __init__(
        self,
        model,
        model_file: str,
        dataset_meta,
        score_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ):
        """Initializes the Inferencer with the model, configuration, and thresholds.

        Raises:
            AssertionError: If the data preprocessor type is not `DetDataPreprocessor`.
        """
        self.model = model
        self.dataset_meta = dataset_meta

        self.cfg = Config.fromfile(model_file)
        test_cfg = self.cfg.model.test_cfg[0]  # the 0th test_cfg is for the query_head
        self.score_threshold = test_cfg.get("score_thr", 0)
        if score_threshold is not None:
            self.score_threshold = score_threshold
        self.with_nms = False
        if "nms" in test_cfg:
            self.with_nms = True
            self.iou_threshold = test_cfg["nms"].get("iou_threshold", 0.8)
            if iou_threshold is not None:
                self.iou_threshold = iou_threshold

        assert self.cfg.model.data_preprocessor.pop("type") == "DetDataPreprocessor"
        # since the image will already be loaded in RGB, it's not necessary to reorder the channels from BGR -> RGB
        self.cfg.model.data_preprocessor["bgr_to_rgb"] = False
        self.data_preprocessor = DetDataPreprocessor(**self.cfg.model.data_preprocessor)

        self.pipeline = self._init_pipeline(self.cfg)
        self.collate_fn = pseudo_collate
        self.visualizer = self._init_visualizer(self.cfg)
        self.num_visualized_imgs = 0
        self.num_predicted_imgs = 0

    def _init_pipeline(self, cfg: Config) -> Compose:
        """Initialize the test pipeline.

        Args:
            cfg (Config): The model configuration.

        Returns:
            Compose: A composed pipeline for preprocessing input data.

        Raises:
            ValueError: If `LoadImageFromFile` is not found in the pipeline configuration.
        """
        pipeline_cfg = cfg.test_dataloader.dataset.pipeline

        # For inference, the key of ``img_id`` is not used.
        if "meta_keys" in pipeline_cfg[-1]:
            pipeline_cfg[-1]["meta_keys"] = tuple(
                meta_key for meta_key in pipeline_cfg[-1]["meta_keys"] if meta_key != "img_id"
            )

        load_img_idx = self._get_transform_idx(pipeline_cfg, ("LoadImageFromFile", LoadImageFromFile))
        if load_img_idx == -1:
            raise ValueError("LoadImageFromFile is not found in the test pipeline")
        pipeline_cfg[load_img_idx]["type"] = "mmdet.InferencerLoader"
        return Compose(pipeline_cfg)

    def _get_transform_idx(self, pipeline_cfg: Config, name: Union[str, Tuple[str, type]]) -> int:
        """Returns the index of a specific transform in the pipeline.

        Args:
            pipeline_cfg (Config): The pipeline configuration.
            name (Union[str, Tuple[str, type]]): The name or type of the transform to find.

        Returns:
            int: The index of the transform in the pipeline. Returns -1 if not found.
        """
        for i, transform in enumerate(pipeline_cfg):
            if transform["type"] in name:
                return i
        return -1

    def _init_visualizer(self, cfg: Config) -> Optional[Visualizer]:
        """Initialize the visualizer for predictions.

        Args:
            cfg (Config): The model configuration.

        Returns:
            Optional[Visualizer]: A visualizer instance or None if not configured.

        Raises:
            ValueError: If the visualizer is not defined in the configuration.
        """
        if "visualizer" not in cfg:
            return None
        timestamp = str(datetime.timestamp(datetime.now()))
        name = cfg.visualizer.get("name", timestamp)
        if Visualizer.check_instance_created(name):
            name = f"{name}-{timestamp}"
        cfg.visualizer.name = name
        visualizer = VISUALIZERS.build(cfg.visualizer)
        visualizer.dataset_meta = self.dataset_meta
        return visualizer

    def add_pred_to_datasample(self, data_samples, results_list):
        """Add predictions to `DetDataSample`.

        Args:
            data_samples (list[DetDataSample]): A batch of data samples containing annotations.
            results_list (list[InstanceData]): Detection results for each image.

        Returns:
            list[DetDataSample]: Updated data samples with predictions added.
        """
        for data_sample, pred_instances in zip(data_samples, results_list):
            data_sample.pred_instances = pred_instances
        samplelist_boxtype2tensor(data_samples)
        return data_samples

    def visualize(
        self,
        inputs: List[np.ndarray],
        preds: SampleList,
        return_vis: bool = False,
        show: bool = False,
        wait_time: int = 0,
        draw_pred: bool = True,
        pred_score_thr: float = 0.3,
        no_save_vis: bool = False,
        img_out_dir: str = "",
        **kwargs,
    ) -> Union[List[np.ndarray], None]:
        """Visualize predictions on input images.

        Args:
            inputs (List[np.ndarray]): List of input images.
            preds (SampleList): Predictions for the input images.
            return_vis (bool): Whether to return the visualization results. Default: False.
            show (bool): Whether to display the images in a popup window. Default: False.
            wait_time (int): Time to wait between displaying images (in seconds). Default: 0.
            draw_pred (bool): Whether to draw predicted bounding boxes. Default: True.
            pred_score_thr (float): Minimum score threshold for drawing predictions. Default: 0.3.
            no_save_vis (bool): Whether to disable saving visualization results. Default: False.
            img_out_dir (str): Directory to save visualization results. Default: "".

        Returns:
            Union[List[np.ndarray], None]: Visualization results if applicable, otherwise None.

        Raises:
            ValueError: If the visualizer is not defined in the configuration.
        """
        if no_save_vis is True:
            img_out_dir = ""

        if not show and img_out_dir == "" and not return_vis:
            return None

        if self.visualizer is None:
            raise ValueError('Visualization needs the "visualizer" term' "defined in the config, but got None.")

        results = []

        for single_input, pred in zip(inputs, preds):
            if isinstance(single_input, str):
                img_bytes = mmengine.fileio.get(single_input)
                img = mmcv.imfrombytes(img_bytes)
                img = img[:, :, ::-1]
                img_name = osp.basename(single_input)
            elif isinstance(single_input, np.ndarray):
                img = single_input.copy()
                img_num = str(self.num_visualized_imgs).zfill(8)
                img_name = f"{img_num}.jpg"
            else:
                raise ValueError("Unsupported input type: " f"{type(single_input)}")

            out_file = osp.join(img_out_dir, "vis", img_name) if img_out_dir != "" else None

            self.visualizer.add_datasample(
                img_name,
                img,
                pred,
                show=show,
                wait_time=wait_time,
                draw_gt=False,
                draw_pred=draw_pred,
                pred_score_thr=pred_score_thr,
                out_file=out_file,
            )
            results.append(self.visualizer.get_image())
            self.num_visualized_imgs += 1

        return results

    def postprocess(
        self,
        preds: SampleList,
        visualization: Optional[List[np.ndarray]] = None,
        return_datasamples: bool = False,
        print_result: bool = False,
        no_save_pred: bool = False,
        pred_out_dir: str = "",
        **kwargs,
    ) -> Dict:
        """Process the predictions and visualization results from ``forward``
        and ``visualize``.

        This method should be responsible for the following tasks:

        1. Convert datasamples into a json-serializable dict if needed.
        2. Pack the predictions and visualization results and return them.
        3. Dump or log the predictions.

        Args:
            preds (List[:obj:`DetDataSample`]): Predictions of the model.
            visualization (Optional[np.ndarray]): Visualized predictions.
            return_datasamples (bool): Whether to use Datasample to store
                inference results. If False, dict will be used.
            print_result (bool): Whether to print the inference result w/o
                visualization to the console. Defaults to False.
            no_save_pred (bool): Whether to force not to save prediction
                results. Defaults to False.
            pred_out_dir: Dir to save the inference results w/o
                visualization. If left as empty, no file will be saved.
                Defaults to ''.

        Returns:
            dict: Inference and visualization results with key ``predictions``
            and ``visualization``.

            - ``visualization`` (Any): Returned by :meth:`visualize`.
            - ``predictions`` (dict or DataSample): Returned by
                :meth:`forward` and processed in :meth:`postprocess`.
                If ``return_datasamples=False``, it usually should be a
                json-serializable dict containing only basic data elements such
                as strings and numbers.
        """
        if no_save_pred is True:
            pred_out_dir = ""

        result_dict = {}
        results = preds
        if not return_datasamples:
            results = []
            for pred in preds:
                result = self.pred2dict(pred, pred_out_dir)
                results.append(result)
        elif pred_out_dir != "":
            warnings.warn(
                "Currently does not support saving datasample "
                "when return_datasamples is set to True. "
                "Prediction results are not saved!"
            )
        # Add img to the results after printing and dumping
        result_dict["predictions"] = results
        if print_result:
            print(result_dict)
        result_dict["visualization"] = visualization
        return result_dict

    def pred2dict(self, data_sample: DetDataSample, pred_out_dir: str = "") -> Dict:
        """Extract elements necessary to represent a prediction into a
        dictionary.

        It's better to contain only basic data elements such as strings and
        numbers in order to guarantee it's json-serializable.

        Args:
            data_sample (:obj:`DetDataSample`): Predictions of the model.
            pred_out_dir: Dir to save the inference results w/o
                visualization. If left as empty, no file will be saved.
                Defaults to ''.

        Returns:
            dict: Prediction results.
        """
        is_save_pred = True
        if pred_out_dir == "":
            is_save_pred = False

        if is_save_pred and "img_path" in data_sample:
            img_path = osp.basename(data_sample.img_path)
            img_path = osp.splitext(img_path)[0]
            out_img_path = osp.join(pred_out_dir, "preds", img_path + "_panoptic_seg.png")
            out_json_path = osp.join(pred_out_dir, "preds", img_path + ".json")
        elif is_save_pred:
            out_img_path = osp.join(pred_out_dir, "preds", f"{self.num_predicted_imgs}_panoptic_seg.png")
            out_json_path = osp.join(pred_out_dir, "preds", f"{self.num_predicted_imgs}.json")
            self.num_predicted_imgs += 1

        result = {}
        if "pred_instances" in data_sample:
            masks = data_sample.pred_instances.get("masks")
            pred_instances = data_sample.pred_instances.numpy()
            result = {"labels": pred_instances.labels.tolist(), "scores": pred_instances.scores.tolist()}
            if "bboxes" in pred_instances:
                result["bboxes"] = pred_instances.bboxes.tolist()
            if masks is not None:
                if "bboxes" not in pred_instances or pred_instances.bboxes.sum() == 0:
                    # Fake bbox, such as the SOLO.
                    bboxes = mask2bbox(masks.cpu()).numpy().tolist()
                    result["bboxes"] = bboxes
                encode_masks = encode_mask_results(pred_instances.masks)
                for encode_mask in encode_masks:
                    if isinstance(encode_mask["counts"], bytes):
                        encode_mask["counts"] = encode_mask["counts"].decode()
                result["masks"] = encode_masks

        if is_save_pred:
            mmengine.dump(result, out_json_path)

        return result

    def run_inference(self, batch_inputs: torch.Tensor, batch_data_samples: SampleList) -> InstanceList:
        """Run inference on a batch of inputs.

        Args:
            batch_inputs (torch.Tensor): Input tensor of shape `(bs, dim, H, W)`.
            batch_data_samples (SampleList): List of data samples for the batch.

        Returns:
            InstanceList: List of detection results for each image in the batch.
        """
        bs, _, H, W = batch_inputs.shape
        # 0 within image, 1 in padded region
        img_masks = torch.ones((bs, H, W), device=batch_inputs.device, dtype=batch_inputs.dtype)
        for i, data_samples in enumerate(batch_data_samples):
            unpad_h, unpad_w = data_samples.metainfo.get("img_unpadded_shape", (H, W))
            img_masks[i, :unpad_h, :unpad_w] = 0
        # predictions is a tuple of (boxes, scores, labels)
        # boxes.shape (bs,max_per_img,4)
        # scores.shape (bs,max_per_img)
        # labels.shape (bs,max_per_img)
        # where max_per_img is the maximum number of detections per image
        predictions = self.model(batch_inputs, img_masks)
        pp_predictions = self.postprocess_predictions(*predictions)
        results_list = []
        for i, (boxes, scores, labels) in enumerate(pp_predictions):

            # rescale to the original image size
            scale_factor = batch_data_samples[i].metainfo["scale_factor"]
            boxes /= boxes.new_tensor(scale_factor).repeat((1, 2))

            results = InstanceData()
            results.bboxes = boxes
            results.scores = scores
            results.labels = labels
            results_list.append(results)
        return results_list

    def postprocess_predictions(
        self, batch_boxes: torch.Tensor, batch_scores: torch.Tensor, batch_labels: torch.Tensor
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        TensorRT requires static output shapes so the score thresholding and
        batch_nms had to be moved outside of the model to the post-processing stage
        """
        processed_predictions = []
        for boxes, scores, labels in zip(batch_boxes, batch_scores, batch_labels):
            if self.score_threshold > 0:
                valid_mask = scores > self.score_threshold
                # TODO: handle case where valid_mask is all False
                scores = scores[valid_mask]
                boxes = boxes[valid_mask]
                labels = labels[valid_mask]

            if self.with_nms:
                keep_idxs = batched_nms(boxes, scores, labels, self.iou_threshold)
                boxes = boxes[keep_idxs]
                scores = scores[keep_idxs]
                labels = labels[keep_idxs]
            # TODO: handle case where det_bboxes is empty

            processed_predictions.append((boxes, scores, labels))
        return processed_predictions

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
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float32,
    ):
        """Run inference on a list of images.

        Args:
            images (List[np.ndarray]): List of input images in RGB format.
            return_vis (bool): Whether to return visualization results. Default: False.
            show (bool): Whether to display images in a popup window. Default: False.
            wait_time (int): Time to wait between displaying images (in seconds). Default: 0.
            no_save_vis (bool): Whether to disable saving visualization results. Default: False.
            draw_pred (bool): Whether to draw predicted bounding boxes. Default: True.
            pred_score_thr (float): Minimum score threshold for drawing predictions. Default: 0.3.
            return_datasamples (bool): Whether to return predictions as `DetDataSample`. Default: False.
            print_result (bool): Whether to print predictions to the console. Default: False.
            no_save_pred (bool): Whether to disable saving predictions. Default: True.
            out_dir (str): Directory to save predictions and visualizations. Default: "".
            device (str): Device to run inference on. Default: "cuda:0".
            dtype (torch.dtype): Data type for inference. Default: `torch.float32`.

        Returns:
            Dict: A dictionary containing predictions and visualization results.
        """
        results_dict = {"predictions": [], "visualization": []}
        for image in images:
            data = self.pipeline(image)  # dict containing keys "inputs" and "data_samples"
            data = self.collate_fn([data])  # dict containing keys "inputs" and "data_samples", where each is a list

            # ori_imgs are the numpy images in the batch, which in this case is just one image
            # data is dict containing keys "inputs" and "data_samples"
            # data["inputs"] is a list of torch tensors of shape (3, H, W)
            # data["data_samples"] is a list of DetDataSample objects
            with torch.no_grad():
                data_processed = self.data_preprocessor(data, False)
                batch_inputs = data_processed["inputs"].to(device).to(dtype)
                batch_data_samples = data_processed["data_samples"]
                results_list = self.run_inference(batch_inputs, batch_data_samples)

            # InstanceList -> SampleList
            preds = self.add_pred_to_datasample(batch_data_samples, results_list)

            visualization = self.visualize(
                [
                    image,
                ],
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
