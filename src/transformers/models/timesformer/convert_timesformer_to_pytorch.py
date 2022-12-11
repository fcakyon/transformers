# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert TimeSformer checkpoints from the original repository: https://github.com/facebookresearch/TimeSformer"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from huggingface_hub import hf_hub_download
from transformers import TimesformerConfig, TimesformerForVideoClassification, VideoMAEImageProcessor

HF_WRITE_TOKEN = "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"


ORIGINAL_FILENAME_TO_INFO = {
    "TimeSformer_divST_8x32_224_K400.pyth": {
        "hf_name": "timesformer-base-finetuned-k400",
        "expected_output_shape": torch.Size([1, 400]),
        "expected_top5_pred_ids": [5, 356, 240, 134, 219],
    },
    "TimeSformer_divST_96x4_224_K400.pyth": {
        "hf_name": "timesformer-large-finetuned-k400",
        "expected_output_shape": torch.Size([1, 400]),
        "expected_top5_pred_ids": [5, 134, 356, 142, 87],
    },
    "TimeSformer_divST_16x16_448_K400.pyth": {
        "hf_name": "timesformer-hr-finetuned-k400",
        "expected_output_shape": torch.Size([1, 400]),
        "expected_top5_pred_ids": [5, 356, 134, 219, 240],
    },
    "TimeSformer_divST_8x32_224_K600.pyth": {
        "hf_name": "timesformer-base-finetuned-k600",
        "expected_output_shape": torch.Size([1, 600]),
        "expected_top5_pred_ids": [9, 189, 527, 198, 74],
    },
    "TimeSformer_divST_96x4_224_K600.pyth": {
        "hf_name": "timesformer-large-finetuned-k600",
        "expected_output_shape": torch.Size([1, 600]),
        "expected_top5_pred_ids": [9, 527, 189, 305, 366],
    },
    "TimeSformer_divST_16x16_448_K600.pyth": {
        "hf_name": "timesformer-hr-finetuned-k600",
        "expected_output_shape": torch.Size([1, 600]),
        "expected_top5_pred_ids": [9, 527, 327, 305, 74],
    },
    "TimeSformer_divST_8_224_SSv2.pyth": {
        "hf_name": "timesformer-base-finetuned-ssv2",
        "expected_output_shape": torch.Size([1, 174]),
        "expected_top5_pred_ids": [56, 17, 125, 15, 126],
    },
    "TimeSformer_divST_64_224_SSv2.pyth": {
        "hf_name": "timesformer-large-finetuned-ssv2",
        "expected_output_shape": torch.Size([1, 174]),
        "expected_top5_pred_ids": [15, 139, 14, 146, 151],
    },
    "TimeSformer_divST_16_448_SSv2.pyth": {
        "hf_name": "timesformer-hr-finetuned-ssv2",
        "expected_output_shape": torch.Size([1, 174]),
        "expected_top5_pred_ids": [98, 112, 160, 56, 66],
    },
}


def get_timesformer_config(model_name):
    config = TimesformerConfig()

    if "large" in model_name and "ssv2" not in model_name:
        config.num_frames = 96
        config.image_size = 224
    elif "large" in model_name and "ssv2" in model_name:
        config.num_frames = 64
        config.image_size = 224
    elif "hr" in model_name:
        config.num_frames = 16
        config.image_size = 448
    else:
        config.num_frames = 8
        config.image_size = 224

    if "k400" in model_name:
        config.num_labels = 400
        repo_id = "huggingface/label-files"
        filename = "kinetics400-id2label.json"
        id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    elif "k600" in model_name:
        config.num_labels = 600
        filename = "kinetics600-id2label.json"
        repo_id = "fcakyon/label-files"
        filename = "kinetics600-id2label.json"
        id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    elif "ssv2" in model_name:
        config.num_labels = 174
        repo_id = "huggingface/label-files"
        filename = "something-something-v2-id2label.json"
        id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    else:
        raise ValueError("Model name should either contain 'k400', 'k600' or 'ssv2'.")

    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}

    return config


def rename_key(name):
    if "encoder." in name:
        name = name.replace("encoder.", "")
    if "cls_token" in name:
        name = name.replace("cls_token", "timesformer.embeddings.cls_token")
    if "pos_embed" in name:
        name = name.replace("pos_embed", "timesformer.embeddings.position_embeddings")
    if "time_embed" in name:
        name = name.replace("time_embed", "timesformer.embeddings.time_embeddings")
    if "patch_embed.proj" in name:
        name = name.replace("patch_embed.proj", "timesformer.embeddings.patch_embeddings.projection")
    if "patch_embed.norm" in name:
        name = name.replace("patch_embed.norm", "timesformer.embeddings.norm")
    if "blocks" in name:
        name = name.replace("blocks", "timesformer.encoder.layer")
    if "attn.proj" in name:
        name = name.replace("attn.proj", "attention.output.dense")
    if "attn" in name and "bias" not in name and "temporal" not in name:
        name = name.replace("attn", "attention.self")
    if "attn" in name and "temporal" not in name:
        name = name.replace("attn", "attention.attention")
    if "temporal_norm1" in name:
        name = name.replace("temporal_norm1", "temporal_layernorm")
    if "temporal_attn.proj" in name:
        name = name.replace("temporal_attn", "temporal_attention.output.dense")
    if "temporal_fc" in name:
        name = name.replace("temporal_fc", "temporal_dense")
    if "norm1" in name and "temporal" not in name:
        name = name.replace("norm1", "layernorm_before")
    if "norm2" in name:
        name = name.replace("norm2", "layernorm_after")
    if "mlp.fc1" in name:
        name = name.replace("mlp.fc1", "intermediate.dense")
    if "mlp.fc2" in name:
        name = name.replace("mlp.fc2", "output.dense")
    if "norm.weight" in name and "fc" not in name and "temporal" not in name:
        name = name.replace("norm.weight", "timesformer.layernorm.weight")
    if "norm.bias" in name and "fc" not in name and "temporal" not in name:
        name = name.replace("norm.bias", "timesformer.layernorm.bias")
    if "head" in name:
        name = name.replace("head", "classifier")

    return name


def convert_state_dict(orig_state_dict, config):
    for key in orig_state_dict.copy().keys():
        val = orig_state_dict.pop(key)

        if key.startswith("model."):
            key = key.replace("model.", "")

        if "qkv" in key:
            key_split = key.split(".")
            layer_num = int(key_split[1])
            prefix = "timesformer.encoder.layer."
            if "temporal" in key:
                postfix = ".temporal_attention.attention.qkv."
            else:
                postfix = ".attention.attention.qkv."
            if "weight" in key:
                orig_state_dict[f"{prefix}{layer_num}{postfix}weight"] = val
            else:
                orig_state_dict[f"{prefix}{layer_num}{postfix}bias"] = val
        else:
            orig_state_dict[rename_key(key)] = val

    return orig_state_dict


def sample_frames_from_video_file(file_path: str, num_frames: int = 16, frame_sampling_rate=1):
    from decord import VideoReader

    videoreader = VideoReader(file_path)
    videoreader.seek(0)

    # sample frames
    start_idx = 0
    end_idx = num_frames * frame_sampling_rate - 1
    indices = np.linspace(start_idx, end_idx, num=num_frames, dtype=np.int64)
    frames = videoreader.get_batch(indices).asnumpy()

    return frames


# We will verify our results on a video of eating spaghetti
def prepare_video(num_frames: int = 8):
    video_path = hf_hub_download(repo_id="nateraw/video-demo", filename="archery.mp4", repo_type="dataset")
    frames = sample_frames_from_video_file(video_path, num_frames=num_frames, frame_sampling_rate=2)
    return list(frames)


def convert_timesformer_checkpoint(original_weight_path, pytorch_dump_folder_path, model_info, push_to_hub=False):
    model_name = model_info["hf_name"]

    config = get_timesformer_config(model_name)

    model = TimesformerForVideoClassification(config)

    state_dict = torch.load(original_weight_path, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
    elif "module" in state_dict:
        state_dict = state_dict["module"]
    else:
        state_dict = state_dict["model_state"]
    new_state_dict = convert_state_dict(state_dict, config)

    model.load_state_dict(new_state_dict)
    model.eval()

    # set processor
    image_processor = VideoMAEImageProcessor(
        image_mean=[0.45, 0.45, 0.45],
        image_std=[0.225, 0.225, 0.225],
        size={"shortest_edge": 448} if "hr" in model_name else {"shortest_edge": 224},
        crop_size={"height": 448, "width": 448} if "hr" in model_name else {"height": 224, "width": 224},
    )

    # verify model on basic input
    video = prepare_video(num_frames=config.num_frames)
    inputs = image_processor(video, return_tensors="pt")

    outputs = model(**inputs)
    logits = outputs.logits
    assert model_info["expected_output_shape"] == logits.shape

    # verify predictions
    topk_probs = logits.softmax(dim=-1).topk(5, dim=-1)
    topk_labels = [model.config.id2label[idx] for idx in topk_probs.indices[0].tolist()]
    assert model_info["expected_top5_pred_ids"] == topk_probs.indices[0].tolist()

    if pytorch_dump_folder_path is not None:
        print(f"Saving model and processor to {pytorch_dump_folder_path}")

        image_processor.save_pretrained(pytorch_dump_folder_path)
        model.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        print("Pushing to the hub...")
        model.push_to_hub(f"fcakyon/{model_name}", use_auth_token=HF_WRITE_TOKEN)
        image_processor.push_to_hub(f"fcakyon/{model_name}", use_auth_token=HF_WRITE_TOKEN)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--weights_folder_dir",
        default=".",
        type=str,
        help=("Folder that contains the TimeSformer weigh files."),
    )
    parser.add_argument(
        "--output_folder_dir",
        default="timesformer_exports/",
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )

    args = parser.parse_args()

    files = os.listdir(args.weights_folder_dir)
    weight_files = [file for file in files if file.endswith(".pyth")]

    if len(weight_files) == 0:
        raise ValueError(
            "The weights_folder_dir should contain at least one file."
            "Please check that the folder exists and contains the weights files."
        )

    for file in files:
        if file.endswith(".pyth"):
            model_name = ORIGINAL_FILENAME_TO_INFO[file]["hf_name"]
            convert_timesformer_checkpoint(
                original_weight_path=os.path.join(args.weights_folder_dir, file),
                pytorch_dump_folder_path=os.path.join(args.output_folder_dir, model_name),
                model_info=ORIGINAL_FILENAME_TO_INFO[file],
                push_to_hub=args.push_to_hub,
            )
