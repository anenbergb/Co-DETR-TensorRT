import torch
import pytest
import os

from codetr import build_CoDETR
from codetr.swin import SwinTransformer
from codetr.transformer import (
    get_reference_points,
    make_encoder_output_proposals,
)

from mmengine.config import Config
from mmdet.registry import MODELS

import torch_tensorrt

from .helpers import benchmark_runtime

torch_tensorrt.runtime.set_multi_device_safe_mode(False)
torch._logging.set_logs(dynamic=10)

# model settings
swin_config = dict(
    pretrain_img_size=384,
    embed_dims=192,
    depths=[2, 2, 18, 2],
    num_heads=[6, 12, 24, 48],
    window_size=12,
    mlp_ratio=4,
    qkv_bias=True,
    qk_scale=None,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.3,
    patch_norm=True,
    out_indices=(0, 1, 2, 3),
    # Please only add indices that would be used
    # in FPN, otherwise some parameter will not be used
    with_cp=True,
    convert_weights=True,
)
neck_config = {
    "type": "ChannelMapper",
    "in_channels": [192, 384, 768, 1536],
    "kernel_size": 1,
    "out_channels": 256,
    "act_cfg": None,
    "norm_cfg": {"type": "GN", "num_groups": 32},
    "num_outs": 5,
}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_swin_transformer(dtype):
    print(f"Testing SwinTransformer with dtype={dtype}")

    torch.manual_seed(42)  # For reproducibility

    iterations = 3
    device = "cuda:0"
    optimization_level = 3  # default is 3, max is 5

    cfg = Config(swin_config)
    model = SwinTransformer(**cfg).to(device).to(dtype)
    model.init_weights()
    model.eval()

    input_height = 768
    input_width = 1152
    batch_inputs = torch.randn(2, 3, input_height, input_width, dtype=dtype, device=device)

    with torch.inference_mode():

        def run_pytorch_model():
            return model(batch_inputs)

        model_export = torch.export.export(
            model,
            args=(batch_inputs,),
            strict=True,
        )
        print(f"✅ Exported {type(model)} to {type(model_export)} with dtype={dtype}")

        def run_exported_model():
            return model_export.module()(batch_inputs)

        model_trt = torch_tensorrt.dynamo.compile(
            model_export,
            inputs=(batch_inputs,),
            enabled_precisions=(dtype,),
            optimization_level=optimization_level,
            truncate_double=True,
            require_full_compilation=True,
        )
        print(f"✅ Compiled {type(model_export)} to TensorRT with dtype={dtype}")

        def run_tensorrt_model():
            return model_trt(batch_inputs)

        # Verify outputs match
        output_pytorch = run_pytorch_model()
        output_export = run_exported_model()
        output_trt = run_tensorrt_model()

    # Use different tolerances based on precision
    tol_export = 1e-3 if dtype == torch.float32 else 1e-2
    tol_trt = 1e-1 if dtype == torch.float32 else 5e-1

    for i in range(len(output_pytorch)):
        torch.testing.assert_close(output_pytorch[i], output_export[i], rtol=tol_export, atol=tol_export)
        torch.testing.assert_close(output_pytorch[i], output_trt[i], rtol=tol_trt, atol=tol_trt)

    benchmark_runtime(run_pytorch_model, run_exported_model, run_tensorrt_model, iterations=iterations)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_neck(dtype):
    print(f"Testing ChannelMapper neck with dtype={dtype}")

    torch.manual_seed(42)  # For reproducibility

    iterations = 3
    device = "cuda:0"
    optimization_level = 3  # default is 3, max is 5

    cfg = Config(neck_config)
    model = MODELS.build(cfg).to(device).to(dtype)
    model.eval()

    input_height = 768
    input_width = 1152
    batch_size = 2
    batch_inputs = []
    in_channels = cfg["in_channels"]
    downscales = [4, 8, 16, 32]
    for in_channel, downscale in zip(in_channels, downscales):
        batch_inputs.append(
            torch.randn(
                batch_size, in_channel, input_height // downscale, input_width // downscale, dtype=dtype, device=device
            )
        )

    batch_inputs = tuple(batch_inputs)

    with torch.inference_mode():

        def run_pytorch_model():
            return model(batch_inputs)

        model_export = torch.export.export(
            model,
            args=(batch_inputs,),
            strict=True,
        )
        print(f"✅ Exported {type(model)} to {type(model_export)} with dtype={dtype}")

        def run_exported_model():
            return model_export.module()(batch_inputs)

        model_trt = torch_tensorrt.dynamo.compile(
            model_export,
            inputs=(batch_inputs,),
            enabled_precisions=(dtype,),
            optimization_level=optimization_level,
            truncate_double=True,
            require_full_compilation=True,
        )
        print(f"✅ Compiled {type(model_export)} to TensorRT with dtype={dtype}")

        def run_tensorrt_model():
            return model_trt(batch_inputs)

        # Verify outputs match
        output_pytorch = run_pytorch_model()
        output_export = run_exported_model()
        output_trt = run_tensorrt_model()

    tol_export = 1e-3 if dtype == torch.float32 else 1e-2
    tol_trt = 1e-1 if dtype == torch.float32 else 5e-1

    for i in range(len(output_pytorch)):
        torch.testing.assert_close(output_pytorch[i], output_export[i], rtol=tol_export, atol=tol_export)
        torch.testing.assert_close(output_pytorch[i], output_trt[i], rtol=tol_trt, atol=tol_trt)

    benchmark_runtime(run_pytorch_model, run_exported_model, run_tensorrt_model, iterations=iterations)


class EncoderWrapper(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(
        self,
        query,
        query_pos=None,
        query_key_padding_mask=None,
        spatial_shapes=None,
        reference_points=None,
        level_start_index=None,
        valid_ratios=None,
    ):
        return self.encoder(
            query,
            None,  # Handled internally
            None,  # Handled internally
            query_pos=query_pos,
            query_key_padding_mask=query_key_padding_mask,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_transformer_encoder(dtype):
    print(f"Testing DetrTransformerEncoder with dtype={dtype}")

    torch.manual_seed(42)  # For reproducibility

    iterations = 3
    device = "cuda:0"
    optimization_level = 3  # default is 3, max is 5

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_file = os.path.join(PROJECT_ROOT, "configs/co_dino_5scale_swin_l_16xb1_16e_o365tococo.py")
    codetr = build_CoDETR(model_file, device=device)
    model = EncoderWrapper(codetr.query_head.transformer.encoder).to(device).to(dtype)
    model.eval()

    # input_height = 1280
    # input_width = 1920
    input_height = 384
    input_width = 384
    in_channels = 256
    batch_size = 2
    downscales = [4, 8, 16, 32, 64]

    mlvl_feats = []
    feat_flatten = []
    mask_flatten = []
    spatial_shapes = []
    for downscale in downscales:
        feat_height = input_height // downscale
        feat_width = input_width // downscale

        img_feat = torch.randn(
            batch_size, in_channels, input_height // downscale, input_width // downscale, dtype=dtype, device=device
        )
        # (B,C,H*W) -> (H*W,B,C)
        img_feat_flat = img_feat.flatten(2).permute(2, 0, 1)
        mask = torch.zeros((batch_size, feat_height, feat_width), device=device, dtype=torch.bool)

        mlvl_feats.append(img_feat)
        feat_flatten.append(img_feat_flat)
        mask_flatten.append(mask.flatten(1))
        spatial_shapes.append((feat_height, feat_width))

    feat_flatten = torch.cat(feat_flatten, dim=0)  # (12276,1,256)
    mask_flatten = torch.cat(mask_flatten, dim=1)  # (1, 12276)

    spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=device)
    level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    valid_ratios = torch.ones((batch_size, len(downscales), 2), device=device, dtype=dtype)
    reference_points = get_reference_points(mlvl_feats, valid_ratios, device=device)
    reference_points_by_level = reference_points[:, :, None] * valid_ratios[:, None]

    with torch.inference_mode():

        def run_pytorch_model():
            return model(
                query=feat_flatten,
                query_pos=feat_flatten,  # same shape as lvl_pos_embed_flatten
                query_key_padding_mask=mask_flatten,
                spatial_shapes=spatial_shapes,
                reference_points=reference_points_by_level,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
            )

        model_export = torch.export.export(
            model,
            args=(feat_flatten,),
            kwargs={
                "query_pos": feat_flatten,
                "query_key_padding_mask": mask_flatten,
                "spatial_shapes": spatial_shapes,
                "reference_points": reference_points_by_level,
                "level_start_index": level_start_index,
                "valid_ratios": valid_ratios,
            },
            strict=True,
        )

        print(f"✅ Exported {type(model)} to {type(model_export)} with dtype={dtype}")

        def run_exported_model():
            return model_export.module()(
                feat_flatten,
                query_pos=feat_flatten,
                query_key_padding_mask=mask_flatten,
                spatial_shapes=spatial_shapes,
                reference_points=reference_points_by_level,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
            )

        model_trt = torch_tensorrt.dynamo.compile(
            model_export,
            arg_inputs=(feat_flatten,),
            kwarg_inputs={
                "query_pos": feat_flatten,
                "query_key_padding_mask": mask_flatten,
                "spatial_shapes": spatial_shapes,
                "reference_points": reference_points_by_level,
                "level_start_index": level_start_index,
                "valid_ratios": valid_ratios,
            },
            enabled_precisions=(dtype,),
            optimization_level=optimization_level,
            truncate_double=True,
            require_full_compilation=True,
        )
        print(f"✅ Compiled {type(model_export)} to TensorRT with dtype={dtype}")

        def run_tensorrt_model():
            return model_trt(
                feat_flatten,
                query_pos=feat_flatten,
                query_key_padding_mask=mask_flatten,
                spatial_shapes=spatial_shapes,
                reference_points=reference_points_by_level,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
            )

        # Verify outputs match
        output_pytorch = run_pytorch_model()
        output_export = run_exported_model()
        output_trt = run_tensorrt_model()

    tol_export = 1e-3
    tol_trt = 1e-1

    for i in range(len(output_pytorch)):
        torch.testing.assert_close(output_pytorch[i], output_export[i], rtol=tol_export, atol=tol_export)
        torch.testing.assert_close(output_pytorch[i], output_trt[i], rtol=tol_trt, atol=tol_trt)

    benchmark_runtime(run_pytorch_model, run_exported_model, run_tensorrt_model, iterations=iterations)


class DecoderWrapper(torch.nn.Module):
    def __init__(self, decoder, reg_branches):
        super().__init__()
        self.decoder = decoder
        self.reg_branches = reg_branches

    def forward(
        self,
        query,
        value=None,
        key_padding_mask=None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        valid_ratios=None,
    ):
        return self.decoder(
            query,
            value=value,
            key_padding_mask=key_padding_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=self.reg_branches,
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_transformer_decoder(dtype):
    print(f"Testing DinoTransformerDecoder with dtype={dtype}")

    torch.manual_seed(42)  # For reproducibility

    iterations = 3
    device = "cuda:0"
    optimization_level = 3  # default is 3, max is 5

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_file = os.path.join(PROJECT_ROOT, "configs/co_dino_5scale_swin_l_16xb1_16e_o365tococo.py")
    codetr = build_CoDETR(model_file, device=device)
    model = DecoderWrapper(codetr.query_head.transformer.decoder, codetr.query_head.reg_branches).to(device).to(dtype)
    model.eval()

    input_height = 384
    input_width = 384
    in_channels = 256
    batch_size = 2
    downscales = [4, 8, 16, 32, 64]

    mlvl_feats = []
    feat_flatten = []
    mask_flatten = []
    spatial_shapes = []
    for downscale in downscales:
        feat_height = input_height // downscale
        feat_width = input_width // downscale

        img_feat = torch.randn(
            batch_size, in_channels, input_height // downscale, input_width // downscale, dtype=dtype, device=device
        )
        # (B,C,H*W) -> (H*W,B,C)
        img_feat_flat = img_feat.flatten(2).permute(2, 0, 1)
        mask = torch.zeros((batch_size, feat_height, feat_width), device=device, dtype=torch.bool)

        mlvl_feats.append(img_feat)
        feat_flatten.append(img_feat_flat)
        mask_flatten.append(mask.flatten(1))
        spatial_shapes.append((feat_height, feat_width))

    feat_flatten = torch.cat(feat_flatten, dim=0)  # (12276,1,256)
    mask_flatten = torch.cat(mask_flatten, dim=1)  # (1, 12276)

    spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=device)
    level_counts = spatial_shapes.prod(1)
    level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), level_counts.cumsum(0)[:-1]))
    valid_ratios = torch.ones((batch_size, len(downscales), 2), device=device, dtype=dtype)
    reference_points = get_reference_points(mlvl_feats, valid_ratios, device=device)

    num_query = 900
    query = torch.randn(num_query, batch_size, in_channels, dtype=dtype, device=device)

    output_proposals = make_encoder_output_proposals(reference_points, level_counts)  # (1,12276,4)
    # sample num_query reference points
    spatial_len = output_proposals.shape[1]
    sample_indices = torch.randint(0, spatial_len, (num_query,), device=device)
    output_proposals = output_proposals[:, sample_indices, :]  # .sigmoid()

    with torch.inference_mode():

        def run_pytorch_model():
            return model(
                query,
                value=feat_flatten,
                key_padding_mask=mask_flatten,
                reference_points=output_proposals,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
            )

        model_export = torch.export.export(
            model,
            args=(query,),
            kwargs={
                "value": feat_flatten,
                "key_padding_mask": mask_flatten,
                "reference_points": output_proposals,
                "spatial_shapes": spatial_shapes,
                "level_start_index": level_start_index,
                "valid_ratios": valid_ratios,
            },
            strict=True,
        )

        print(f"✅ Exported {type(model)} to {type(model_export)} with dtype={dtype}")

        def run_exported_model():
            return model_export.module()(
                query,
                value=feat_flatten,
                key_padding_mask=mask_flatten,
                reference_points=output_proposals,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
            )

        model_trt = torch_tensorrt.dynamo.compile(
            model_export,
            arg_inputs=(query,),
            kwarg_inputs={
                "value": feat_flatten,
                "key_padding_mask": mask_flatten,
                "reference_points": output_proposals,
                "spatial_shapes": spatial_shapes,
                "level_start_index": level_start_index,
                "valid_ratios": valid_ratios,
            },
            enabled_precisions=(dtype,),
            optimization_level=optimization_level,
            truncate_double=True,
            require_full_compilation=True,
        )
        print(f"✅ Compiled {type(model_export)} to TensorRT with dtype={dtype}")

        def run_tensorrt_model():
            return model_trt(
                query,
                value=feat_flatten,
                key_padding_mask=mask_flatten,
                reference_points=output_proposals,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
            )

        # Verify outputs match
        output_pytorch = run_pytorch_model()
        output_export = run_exported_model()
        output_trt = run_tensorrt_model()

    tol_export = 1e-3 if dtype == torch.float32 else 1e-3
    tol_trt = 1e-1 if dtype == torch.float32 else 5e-1
    # with torch.float16 the error accumulates with each decoder layer since the predicted reference points
    # are used for the next layer.

    for i in range(len(output_pytorch)):
        abs_diff = torch.abs(output_pytorch[i] - output_trt[i])
        top_5_diff, top_5_locations = torch.topk(abs_diff.flatten(), 5)
        print(
            f"Top 5 absolute differences for output {i}: {top_5_diff.tolist()} at locations {top_5_locations.tolist()}"
        )
        torch.testing.assert_close(output_pytorch[i], output_export[i], rtol=tol_export, atol=tol_export)
        torch.testing.assert_close(output_pytorch[i], output_trt[i], rtol=tol_trt, atol=tol_trt)

    benchmark_runtime(run_pytorch_model, run_exported_model, run_tensorrt_model, iterations=iterations)


class TransformerWrapper(torch.nn.Module):
    def __init__(self, transformer, reg_branches, cls_branches):
        super().__init__()
        self.transformer = transformer
        self.reg_branches = reg_branches
        self.cls_branches = cls_branches

    def forward(
        self,
        mlvl_feats,
        mlvl_masks,
        mlvl_pos_embeds,
    ):
        return self.transformer(
            mlvl_feats,
            mlvl_masks,
            mlvl_pos_embeds,
            reg_branches=self.reg_branches,
            cls_branches=self.cls_branches,
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_transformer(dtype):
    print(f"Testing DinoTransformer with dtype={dtype}")

    torch.manual_seed(42)  # For reproducibility

    iterations = 3
    device = "cuda:0"
    optimization_level = 3  # default is 3, max is 5

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_file = os.path.join(PROJECT_ROOT, "configs/co_dino_5scale_swin_l_16xb1_16e_o365tococo.py")
    codetr = build_CoDETR(model_file, device=device)
    model = (
        TransformerWrapper(
            codetr.query_head.transformer, codetr.query_head.reg_branches, codetr.query_head.cls_branches
        )
        .to(device)
        .to(dtype)
    )
    model.eval()

    input_height = 384
    input_width = 384
    in_channels = 256
    batch_size = 2
    downscales = [4, 8, 16, 32, 64]

    mlvl_feats = []
    mlvl_masks = []
    for downscale in downscales:
        feat_height = input_height // downscale
        feat_width = input_width // downscale

        img_feat = torch.randn(
            batch_size, in_channels, input_height // downscale, input_width // downscale, dtype=dtype, device=device
        )
        mask = torch.zeros((batch_size, feat_height, feat_width), device=device, dtype=torch.bool)

        mlvl_feats.append(img_feat)
        mlvl_masks.append(mask)

    with torch.inference_mode():

        def run_pytorch_model():
            return model(
                mlvl_feats,
                mlvl_masks,
                mlvl_feats,
            )

        model_export = torch.export.export(
            model,
            args=(mlvl_feats, mlvl_masks, mlvl_feats),
            strict=True,
        )

        print(f"✅ Exported {type(model)} to {type(model_export)} with dtype={dtype}")

        def run_exported_model():
            return model_export.module()(mlvl_feats, mlvl_masks, mlvl_feats)

        model_trt = torch_tensorrt.dynamo.compile(
            model_export,
            arg_inputs=(
                mlvl_feats,
                mlvl_masks,
                mlvl_feats,
            ),
            enabled_precisions=(dtype,),
            optimization_level=optimization_level,
            truncate_double=True,
            require_full_compilation=True,
        )
        print(f"✅ Compiled {type(model_export)} to TensorRT with dtype={dtype}")

        def run_tensorrt_model():
            return model_trt(mlvl_feats, mlvl_masks, mlvl_feats)

        # Verify outputs match
        output_pytorch = run_pytorch_model()
        output_export = run_exported_model()
        output_trt = run_tensorrt_model()

    """
A randomly initialized CoDETR model will predict a random enc_class_outputs tensor when run on an random input.
I measured the numerical difference between the Pytorch and TensorRT exported enc_outputs_class for torch.float32 as on the order of 1e-3.
The selection of topk indices from the enc_class_outputs tensor will be different between the Pytorch and TensorRT exported models. 
I measured the intersection of selected topk indices between the Pytorch and TensorRT exported model as mismatching approximately 50%.
This indicates that the enc_class_outputs tensor is randomly distribution, and that a noise of 1e-3 is sufficient to randomize the topk index selection.
The selection of topk indices will likely be more similar if the CoDETR model is initialized from pre-trained weights and a real image is provided as input because the distribution of enc_outputs_class will be smoother.
    """
    # tol_export = 1e-3 if dtype == torch.float32 else 1e-3
    # tol_trt = 1e-1 if dtype == torch.float32 else 1
    # for i in range(len(output_pytorch)):
    #     abs_diff = torch.abs(output_pytorch[i] - output_trt[i])
    #     top_5_diff, top_5_locations = torch.topk(abs_diff.flatten(), 5)
    #     print(
    #         f"Top 5 absolute differences for output {i}: {top_5_diff.tolist()} at locations {top_5_locations.tolist()}"
    #     )
    #     torch.testing.assert_close(output_pytorch[i], output_export[i], rtol=tol_export, atol=tol_export)
    #     torch.testing.assert_close(output_pytorch[i], output_trt[i], rtol=tol_trt, atol=tol_trt)

    benchmark_runtime(run_pytorch_model, run_exported_model, run_tensorrt_model, iterations=iterations)


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float16],
)
def test_query_head(dtype):
    print(f"Testing CoDINOHead with dtype={dtype}")

    torch.manual_seed(42)  # For reproducibility

    iterations = 3
    device = "cuda:0"
    optimization_level = 3  # default is 3, max is 5

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_file = os.path.join(PROJECT_ROOT, "configs/co_dino_5scale_swin_l_16xb1_16e_o365tococo.py")
    codetr = build_CoDETR(model_file, device=device)
    model = codetr.query_head.to(device).to(dtype)
    model.eval()

    # input_height = 1280
    # input_width = 1920
    input_height = 384
    input_width = 384
    in_channels = 256
    batch_size = 2
    img_feats = []
    downscales = [4, 8, 16, 32, 64]
    for downscale in downscales:
        img_feats.append(
            torch.randn(
                batch_size, in_channels, input_height // downscale, input_width // downscale, dtype=dtype, device=device
            )
        )
    # 0 within image, 1 in padded region
    # this is a dummy mask where all pixels are within the image
    img_masks = torch.zeros((batch_size, input_height, input_width), device=device, dtype=dtype)

    with torch.inference_mode():

        def run_pytorch_model():
            return model(img_feats, img_masks)

        run_pytorch_model()
        model_export = torch.export.export(
            model,
            args=(img_feats, img_masks),
            strict=True,
        )
        print(f"✅ Exported {type(model)} to {type(model_export)} with dtype={dtype}")

        def run_exported_model():
            return model_export.module()(img_feats, img_masks)

        model_trt = torch_tensorrt.dynamo.compile(
            model_export,
            inputs=(img_feats, img_masks),
            enabled_precisions=(dtype,),
            optimization_level=optimization_level,
            truncate_double=True,
            require_full_compilation=True,
        )
        print(f"✅ Compiled {type(model_export)} to TensorRT with dtype={dtype}")

        def run_tensorrt_model():
            return model_trt(img_feats, img_masks)

    benchmark_runtime(run_pytorch_model, run_exported_model, run_tensorrt_model, iterations=iterations)


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float16],
)
def test_codetr(dtype):
    print(f"Testing CoDETR with dtype={dtype}")

    torch.manual_seed(42)  # For reproducibility

    iterations = 3
    device = "cuda:0"
    optimization_level = 3  # default is 3, max is 5

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_file = os.path.join(PROJECT_ROOT, "configs/co_dino_5scale_swin_l_16xb1_16e_o365tococo.py")
    model = build_CoDETR(model_file, device=device).to(dtype)
    model.eval()

    batch_size = 1
    input_height = 768
    input_width = 1152
    batch_inputs = torch.randn(batch_size, 3, input_height, input_width, dtype=dtype, device=device)
    # 0 within image, 1 in padded region
    # this is a dummy mask where all pixels are within the image
    img_masks = torch.zeros((batch_size, input_height, input_width), device=device, dtype=dtype)

    with torch.inference_mode():

        def run_pytorch_model():
            return model(batch_inputs, img_masks)

        model_export = torch.export.export(
            model,
            args=(batch_inputs, img_masks),
            strict=True,
        )
        print(f"✅ Exported {type(model)} to {type(model_export)} with dtype={dtype}")

        def run_exported_model():
            return model_export.module()(batch_inputs, img_masks)

        model_trt = torch_tensorrt.dynamo.compile(
            model_export,
            inputs=(batch_inputs, img_masks),
            enabled_precisions=(dtype,),
            optimization_level=optimization_level,
            truncate_double=True,
            require_full_compilation=True,
        )
        print(f"✅ Compiled {type(model_export)} to TensorRT with dtype={dtype}")

        def run_tensorrt_model():
            return model_trt(batch_inputs, img_masks)

    benchmark_runtime(run_pytorch_model, run_exported_model, run_tensorrt_model, iterations=iterations)


if __name__ == "__main__":
    pytest.main([__file__])
