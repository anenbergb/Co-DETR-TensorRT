import torch
import pytest
import numpy as np

from codetr.swin import SwinTransformer
from codetr.co_dino_head import CoDINOHead
from codetr.transformer import (
    DetrTransformerEncoder,
    DinoTransformerDecoder,
    get_reference_points,
    get_encoder_output_proposals,
)

from mmengine.config import Config
from mmdet.registry import MODELS

import torch_tensorrt

from .helpers import benchmark_runtime

torch_tensorrt.runtime.set_multi_device_safe_mode(False)


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

query_head_config = {
    #  'type': 'CoDINOHead',
    "num_query": 900,
    "num_classes": 80,
    "in_channels": 2048,
    "as_two_stage": True,
    "dn_cfg": {
        "label_noise_scale": 0.5,
        "box_noise_scale": 0.4,
        "group_cfg": {"dynamic": True, "num_groups": None, "num_dn_queries": 500},
    },
    "transformer": {
        "type": "CoDinoTransformer",
        "with_coord_feat": False,
        "num_co_heads": 2,
        "num_feature_levels": 5,
        "encoder": {
            "type": "DetrTransformerEncoder",
            "num_layers": 6,
            "with_cp": 6,
            "transformerlayers": {
                "type": "BaseTransformerLayer",
                "attn_cfgs": {
                    "type": "MultiScaleDeformableAttention",
                    "embed_dims": 256,
                    "num_levels": 5,
                    "dropout": 0.0,
                },
                "feedforward_channels": 2048,
                "ffn_dropout": 0.0,
                "operation_order": ("self_attn", "norm", "ffn", "norm"),
            },
        },
        "decoder": {
            "type": "DinoTransformerDecoder",
            "num_layers": 6,
            "return_intermediate": True,
            "transformerlayers": {
                "type": "DetrTransformerDecoderLayer",
                "attn_cfgs": [
                    {"type": "MultiheadAttention", "embed_dims": 256, "num_heads": 8, "dropout": 0.0},
                    {"type": "MultiScaleDeformableAttention", "embed_dims": 256, "num_levels": 5, "dropout": 0.0},
                ],
                "feedforward_channels": 2048,
                "ffn_dropout": 0.0,
                "operation_order": ("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
            },
        },
    },
    "positional_encoding": {"type": "SinePositionalEncoding", "num_feats": 128, "temperature": 20, "normalize": True},
    "loss_cls": {"type": "QualityFocalLoss", "use_sigmoid": True, "beta": 2.0, "loss_weight": 1.0},
    "loss_bbox": {"type": "L1Loss", "loss_weight": 5.0},
    "loss_iou": {"type": "GIoULoss", "loss_weight": 2.0},
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

    batch_inputs = torch.randn(1, 3, 1280, 1920, dtype=dtype, device=device)

    def run_pytorch_model():
        return model(batch_inputs)

    with torch.inference_mode():
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
        )
        print(f"✅ Compiled {type(model_export)} to TensorRT with dtype={dtype}")

        def run_tensorrt_model():
            return model_trt(batch_inputs)

        # Verify outputs match
        output_pytorch = run_pytorch_model()
        output_export = run_exported_model()
        output_trt = run_tensorrt_model()

    # Use different tolerances based on precision
    tol_export = 1e-3 if dtype == torch.float32 else 1e-3
    tol_trt = 1e-1 if dtype == torch.float32 else 1e-1

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
    # model.init_weights()
    model.eval()

    input_height = 1280
    input_width = 1920
    batch_size = 1
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

    def run_pytorch_model():
        return model(batch_inputs)

    with torch.inference_mode():
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
        )
        print(f"✅ Compiled {type(model_export)} to TensorRT with dtype={dtype}")

        def run_tensorrt_model():
            return model_trt(batch_inputs)

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


@pytest.mark.parametrize("dtype", [torch.float32])
def test_query_head(dtype):
    print(f"Testing CoDINOHead with dtype={dtype}")

    torch.manual_seed(42)  # For reproducibility

    iterations = 3
    device = "cuda:0"
    optimization_level = 3  # default is 3, max is 5

    cfg = Config(query_head_config)
    model = CoDINOHead(**cfg).to(device).to(dtype)
    model.init_weights()
    model.to(dtype)
    model.eval()

    # input_height = 1280
    # input_width = 1920
    input_height = 384
    input_width = 384
    in_channels = 256
    batch_size = 1
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
    img_masks = torch.zeros((1, input_height, input_width), device=device, dtype=dtype)

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
        )
        print(f"✅ Compiled {type(model_export)} to TensorRT with dtype={dtype}")

        def run_tensorrt_model():
            return model_trt(img_feats, img_masks)

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


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_transformer_encoder(dtype):
    print(f"Testing DetrTransformerEncoder with dtype={dtype}")

    torch.manual_seed(42)  # For reproducibility

    iterations = 3
    device = "cuda:0"
    optimization_level = 3  # default is 3, max is 5

    cfg = Config(query_head_config)
    encoder_cfg = cfg.transformer.encoder
    assert encoder_cfg.pop("type") == "DetrTransformerEncoder"
    model = DetrTransformerEncoder(**encoder_cfg).to(device).to(dtype)
    model.to(dtype)
    model.eval()

    model = EncoderWrapper(model)

    # input_height = 1280
    # input_width = 1920
    input_height = 384
    input_width = 384
    in_channels = 256
    batch_size = 1
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
        mask = torch.zeros((1, feat_height, feat_width), device=device, dtype=torch.bool)

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

    with torch.inference_mode():

        def run_pytorch_model():
            return model(
                query=feat_flatten,
                query_pos=feat_flatten,  # same shape as lvl_pos_embed_flatten
                query_key_padding_mask=mask_flatten,
                spatial_shapes=spatial_shapes,
                reference_points=reference_points,
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
                "reference_points": reference_points,
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
                reference_points=reference_points,
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
                "reference_points": reference_points,
                "level_start_index": level_start_index,
                "valid_ratios": valid_ratios,
            },
            enabled_precisions=(dtype,),
            optimization_level=optimization_level,
        )
        print(f"✅ Compiled {type(model_export)} to TensorRT with dtype={dtype}")

        def run_tensorrt_model():
            return model_trt(
                feat_flatten,
                query_pos=feat_flatten,
                query_key_padding_mask=mask_flatten,
                spatial_shapes=spatial_shapes,
                reference_points=reference_points,
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
    def __init__(self, decoder, num_reg_fcs=6):
        super().__init__()
        self.decoder = decoder

        self.reg_branches = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(decoder.embed_dims, decoder.embed_dims),
                    torch.nn.ReLU(),
                    torch.nn.Linear(decoder.embed_dims, 4),
                )
                for _ in range(num_reg_fcs)
            ]
        )

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

    cfg = Config(query_head_config)
    decoder_cfg = cfg.transformer.decoder
    assert decoder_cfg.pop("type") == "DinoTransformerDecoder"
    decoder = DinoTransformerDecoder(**decoder_cfg).to(device).to(dtype)
    model = DecoderWrapper(decoder, num_reg_fcs=decoder_cfg.num_layers).to(device).to(dtype)
    model.eval()

    # input_height = 1280
    # input_width = 1920
    input_height = 384
    input_width = 384
    in_channels = 256
    batch_size = 1
    downscales = [4, 8, 16, 32, 64]

    mlvl_feats = []
    feat_flatten = []
    mask_flatten = []
    mlvl_masks = []
    spatial_shapes = []
    for downscale in downscales:
        feat_height = input_height // downscale
        feat_width = input_width // downscale

        img_feat = torch.randn(
            batch_size, in_channels, input_height // downscale, input_width // downscale, dtype=dtype, device=device
        )
        # (B,C,H*W) -> (H*W,B,C)
        img_feat_flat = img_feat.flatten(2).permute(2, 0, 1)
        mask = torch.zeros((1, feat_height, feat_width), device=device, dtype=torch.bool)
        mlvl_masks.append(mask)
        mlvl_feats.append(img_feat)

        feat_flatten.append(img_feat_flat)
        mask_flatten.append(mask.flatten(1))
        spatial_shapes.append((feat_height, feat_width))

    feat_flatten = torch.cat(feat_flatten, dim=0)  # (12276,1,256)
    mask_flatten = torch.cat(mask_flatten, dim=1)  # (1, 12276)

    spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=device)
    level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    valid_ratios = torch.ones((batch_size, len(downscales), 2), device=device, dtype=dtype)

    num_query = 900
    query = torch.randn(num_query, batch_size, in_channels, dtype=dtype, device=device)

    # (1,12276,256), (1,12276,4)
    _, output_proposals = get_encoder_output_proposals(
        feat_flatten.permute(1, 0, 2), mask_flatten, mlvl_masks  # (1,12276,256)
    )
    # sample num_query reference points
    spatial_len = output_proposals.shape[1]
    sample_indices = torch.randint(0, spatial_len, (num_query,), device=device)
    reference_points = output_proposals[:, sample_indices, :].sigmoid()

    with torch.inference_mode():

        def run_pytorch_model():
            return model(
                query,
                value=feat_flatten,
                key_padding_mask=mask_flatten,
                reference_points=reference_points,
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
                "reference_points": reference_points,
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
                reference_points=reference_points,
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
                "reference_points": reference_points,
                "spatial_shapes": spatial_shapes,
                "level_start_index": level_start_index,
                "valid_ratios": valid_ratios,
            },
            enabled_precisions=(dtype,),
            optimization_level=optimization_level,
        )
        print(f"✅ Compiled {type(model_export)} to TensorRT with dtype={dtype}")

        def run_tensorrt_model():
            return model_trt(
                query,
                value=feat_flatten,
                key_padding_mask=mask_flatten,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
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


if __name__ == "__main__":
    pytest.main([__file__])
