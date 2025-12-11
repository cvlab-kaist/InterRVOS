import os
import sys
import json
import cv2
import numpy as np
import argparse
import random
import glob
import shutil
import imageio.v2 as imageio
from tqdm import tqdm

import pycocotools.mask as mask_utils

from process_video import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import load_video, get_frame_indices_only

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="processed/multi")
    parser.add_argument("--image_root", type=str, default="datasets/VidOR/JPEGImages_clips")
    parser.add_argument("--mask_root", type=str, default="datasets/VidOR/MaskAnnotationsV2")
    parser.add_argument("--anno_root", type=str, default="datasets/VidOR/Annotations/train")
    parser.add_argument("--vp_type", type=str, default="mask_overlay")
    parser.add_argument("--num_segments", type=int, default=16)
    parser.add_argument("--num_videos", type=int, default=0)
    args = parser.parse_args()

    # PATH CONFIG
    image_root = args.image_root
    mask_root = args.mask_root
    anno_root = args.anno_root

    # VIDEO IDS IN MASK ROOT
    video_to_process = []
    for mask_json in os.listdir(mask_root):
        video_id = mask_json.split("_")[0]
        bin_list = glob.glob(os.path.join(mask_root, f"{video_id}_*.json"))
        num_splits_needed = len(os.listdir(os.path.join(image_root, video_id)))
        if len(bin_list) != num_splits_needed:
            continue
        if video_id not in video_to_process:
            video_to_process.append(video_id)

    pbar = tqdm(video_to_process, dynamic_ncols=True)

    # SAVE VALID OBJECT IDS INFO
    save_valid_obj_ids_path = os.path.join(args.output_dir, "valid_obj_ids.json")
    if os.path.exists(save_valid_obj_ids_path):
        with open(save_valid_obj_ids_path, "r") as f:
            valid_obj_ids_dict = json.load(f)
    else:
        valid_obj_ids_dict = {}

    count = 0
    mask_dict_error = []

    for video_id in pbar:
        pbar.set_description(f'VIDEO {video_id}')

        if (args.num_videos != 0) and (count >= args.num_videos): break
        count += 1
        
        # CHECK IF THE SPLIT IS CORRECT : PROCESSED VIDEOS
        bin_mask_list = sorted(glob.glob(os.path.join(mask_root, f"{video_id}_*.json")))  # 0, 1, 2 ...
        vid_dir_list = sorted(glob.glob(os.path.join(image_root, video_id, "clip_*")))  # 000, 001, 002 ...
        assert len(bin_mask_list) == len(vid_dir_list)

        # CHECK IF THE SPLIT IS CORRECT : FRAME COUNT
        anno_path = glob.glob(os.path.join(anno_root, "**", f"{video_id}.json"))[0]
        with open(anno_path, "r") as f:
            anno_dict = json.load(f)

        # SUB BIN LIST : MAXIMUM FIRST / LAST TWO BINS
        for i in range(len(vid_dir_list)):
            if len(os.listdir(vid_dir_list[i])) < 200:
                vid_dir_list.pop(i)
        sub_vid_dir_list = [vid_dir_list[0]] if len(vid_dir_list) == 1 else ([vid_dir_list[0], vid_dir_list[-1]] if len(vid_dir_list) >= 2 else [])

        # ### DEBUG
        # if video_id not in ["2458411884", "2452889804", "11794662446"]:
        #     continue

        # PROCESS EACH VIDEO CLIP
        for vid_dir in sub_vid_dir_list:
            split_idx = int(os.path.basename(vid_dir).replace("clip_", ""))
            mask_path = bin_mask_list[split_idx]
            assert int(os.path.basename(mask_path).split("_")[-1].replace(".json", "")) == split_idx
            
            try:
                with open(mask_path, "r") as f:
                    ori_mask_dict = json.load(f)
            except:
                mask_dict_error.append(mask_path)
                print(f"MASK DICT ERROR {mask_path}")
                continue
            
            if ori_mask_dict == {}:
                mask_dict_error.append(mask_path)
                print(f"MASK DICT ERROR {mask_path}")
                continue

            # 1000 BIN INTO 500 VIDEO CLIP : first 200~500 frames in each bin
            new_mask_dict = {}
            for obj_id in sorted(ori_mask_dict.keys()):
                info = ori_mask_dict[obj_id]
                masklet = []
                for mask in info["mask"]:
                    if mask is not None:
                        masklet.append(mask)
                obj_video_clip_masklet = masklet[:500]
                new_mask_dict[obj_id] = obj_video_clip_masklet
            num_frames = len(obj_video_clip_masklet)

            # VIDEO CLIP INFOS
            num_objects = len(new_mask_dict)
            obj_list = new_mask_dict.keys()
            width = anno_dict['width']
            height = anno_dict['height']
            frames = sorted(os.listdir(vid_dir))[:num_frames]
            
            # APPEARANCE INFO FOR EACH OBJECT
            first_frame_dict = {str(object_idx): None for object_idx in obj_list}
            last_frame_dict = {str(object_idx): None for object_idx in obj_list}

            obj_id_to_remove = []
            for obj_id in obj_list:
                for frame_idx, frame in enumerate(frames):
                    current_rle_mask = new_mask_dict[obj_id][frame_idx]
                    current_mask = mask_utils.decode(current_rle_mask)
                    if (np.sum(current_mask) != 0) and (current_rle_mask["counts"] != None):  # not empty mask
                        # FIRST & LAST APPEARANCE
                        if first_frame_dict[obj_id] is None:
                            first_frame_dict[obj_id] = frame_idx
                        last_frame_dict[obj_id] = frame_idx
            
                # FILTER OUT OBJECTS APPEAR LESS THEN 10 FRAMES
                if (first_frame_dict[obj_id] is None) or (last_frame_dict[obj_id] is None):
                    obj_id_to_remove.append(obj_id)
                elif (last_frame_dict[obj_id] - first_frame_dict[obj_id] <= 10):
                    obj_id_to_remove.append(obj_id)
            
            for obj_id in obj_id_to_remove:
                del new_mask_dict[obj_id]
            if len(new_mask_dict) == 0:
                continue
            
            start_frame_idx, end_frame_idx = len(frames), 0
            for obj_id in new_mask_dict.keys():
                if first_frame_dict[obj_id] < start_frame_idx:
                    start_frame_idx = first_frame_dict[obj_id]
                if last_frame_dict[obj_id] > end_frame_idx:
                    end_frame_idx = last_frame_dict[obj_id]

            num_objects = len(new_mask_dict.keys())

            # COLOR PALETTE
            palette = generate_fixed_palette(N=num_objects)
            color_bgrs = [tuple(reversed(rgb)) for _, rgb in palette]

            # PREPARE FOR VIDEO SAVING
            out_video_dir = os.path.join(args.output_dir, f"{video_id}_clip{split_idx:03d}")
            if os.path.exists(out_video_dir) and (len(os.listdir(out_video_dir)) == args.num_segments):
                continue
            os.makedirs(out_video_dir, exist_ok=True)
            
            # CROPPED FRAMES & MASKLET
            crop_frames = frames[start_frame_idx : end_frame_idx+1]
            crop_rle_masklet = {}
            for obj_id in new_mask_dict.keys():
                crop_rle_masklet[obj_id] = new_mask_dict[obj_id][start_frame_idx : end_frame_idx+1]

            # # SAVE CROPPED VIDEOS
            # crop_video_path = "temp_multi.mp4"
            # writer = imageio.get_writer(crop_video_path, fps=30, codec='libx264')
            # for frame in crop_frames:
            #     img_path = os.path.join(vid_dir, frame)
            #     img = cv2.imread(img_path)
            #     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #     writer.append_data(img_rgb)
            # writer.close()
            
            # # FRAMES THAT MLLM SEES
            # _, _, frame_indices = load_video(crop_video_path, num_segments=args.num_segments, return_indices=True)

            # FRAMES THAT MLLM SEES
            frame_indices = get_frame_indices_only(len(crop_frames) - 1, num_segments=args.num_segments)

            # OBTAIN FRAMES THAT MLLM SEES
            sub_frames = []
            for frame_idx in frame_indices:
                frame_idx = int(frame_idx)
                sub_frames.append(crop_frames[frame_idx])
            sub_rle_masklet = {}
            for obj_id in new_mask_dict.keys():
                sub_rle_masklet[obj_id] = []
                for frame_idx in frame_indices:
                    sub_rle_masklet[obj_id].append(crop_rle_masklet[obj_id][frame_idx])

            # DECODE MASKLET
            masklet = {}
            for obj_id, rle_masklet in sub_rle_masklet.items():
                masklet[obj_id] = []
                for frame_idx, rle_mask in enumerate(rle_masklet):
                    if rle_mask is not None:
                        mask = mask_utils.decode(rle_mask).astype(np.uint8)
                    else:
                        mask = np.zeros((height, width)).astype(np.uint8)
                    masklet[obj_id].append(mask*255)
                
            # FILTER OBJECTS APPEAR LESS THAN THRESHOLD
            non_valid_obj_ids, valid_obj_ids = [], []
            for obj_id in new_mask_dict.keys():
                object_ratios = []
                for frame_idx, frame in enumerate(sub_frames):
                    mask = masklet[obj_id][frame_idx]
                    object_area = int(mask.sum() / 255)
                    total_area = mask.size
                    object_ratio = object_area / total_area
                    object_ratios.append(object_ratio)
                zero_count = object_ratios.count(0.0)
                non_zero_count = args.num_segments - zero_count
                if len(new_mask_dict.keys()) > 3:
                    if (max(object_ratios) <= 0.01) or (non_zero_count <= 2):
                        non_valid_obj_ids.append(obj_id)
                    else:
                        valid_obj_ids.append(obj_id)
                else:
                    valid_obj_ids.append(obj_id)
            valid_obj_ids_dict[f"{video_id}_clip{split_idx:03d}"] = valid_obj_ids
            
            # PASS THE VIDEO IF THERE IS NO VALID OBJECTS
            if valid_obj_ids == []:
                shutil.rmtree(out_video_dir)
                continue

            # PROCESS VIDEO : FRAME-WISE PROCESSING
            video_frames_list = []
            for frame_idx, frame_name in enumerate(sub_frames):
                frame_path = os.path.join(vid_dir, frame_name)
                frame = cv2.imread(frame_path)
                
                multi_object_masklets = []
                for obj_id in valid_obj_ids:
                    # MASK
                    rle_mask = sub_rle_masklet[obj_id][frame_idx]
                    mask = mask_utils.decode(rle_mask).astype(np.uint8) * 255 if rle_mask else np.zeros((height, width), np.uint8)
                    multi_object_masklets.append(mask)

                if valid_obj_ids != []:
        
                    # MASK OUTLINE + LABEL
                    if args.vp_type == "":
                        # RESIZE TALL AND HEIGHT MORE THAN 1000 PIXELS
                        if (height > width) and height >= 1000:
                            frame = cv2.resize(frame, (width // 2, height // 2), interpolation=cv2.INTER_AREA)
                            for masklet_idx, masklet in enumerate(multi_object_masklets):
                                multi_object_masklets[masklet_idx] = cv2.resize(multi_object_masklets[masklet_idx], (width // 2, height // 2), interpolation=cv2.INTER_NEAREST)
                        processed_frame = mask_outline_on_frame(frame, masklets=multi_object_masklets, palette=color_bgrs, label=True)
                    
                    # MASK OVERLAY + LABEL
                    elif args.vp_type == "mask_overlay":
                        # RESIZE TALL AND HEIGHT MORE THAN 1000 PIXELS
                        if (height > width) and height >= 1000:
                            frame = cv2.resize(frame, (width // 2, height // 2), interpolation=cv2.INTER_AREA)
                            for masklet_idx, masklet in enumerate(multi_object_masklets):
                                multi_object_masklets[masklet_idx] = cv2.resize(multi_object_masklets[masklet_idx], (width // 2, height // 2), interpolation=cv2.INTER_NEAREST)
                        processed_frame = mask_overlay_on_frame(frame, masklets=multi_object_masklets, palette=color_bgrs, valid_obj_ids=valid_obj_ids, label=True)
        
                    processed_frame = resize_to_even(processed_frame)
                    # SAVE IMAGE
                    save_path = os.path.join(out_video_dir, f"{frame_idx:06d}.jpg")
                    cv2.imwrite(save_path, processed_frame)
        
            # SAVE VALID OBJECT IDS
            with open(save_valid_obj_ids_path, "w") as f:
                json.dump(valid_obj_ids_dict, f, indent=4)

    print("*** ERROR IN MASK DICT ***")
    print(mask_dict_error)
