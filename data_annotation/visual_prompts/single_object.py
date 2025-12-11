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
    parser.add_argument("--output_dir", type=str, default="processed/single")
    parser.add_argument("--image_root", type=str, default="datasets/VidOR/JPEGImages_clips")
    parser.add_argument("--mask_root", type=str, default="datasets/VidOR/MaskAnnotationsV3")
    parser.add_argument("--anno_root", type=str, default="datasets/VidOR/Annotations/train")
    parser.add_argument("--vp_type", type=str, default="")
    parser.add_argument("--num_segments", type=int, default=16)
    parser.add_argument("--num_videos", type=int, default=0)
    args = parser.parse_args()

    # PATH CONFIG
    image_root = args.image_root
    mask_root = args.mask_root
    anno_root = args.anno_root
    
    # VIDEOS DO NOT MATCH WITH FRAMES AND ANNOTATIONS
    video_do_not_match = []

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
        
        ### DEBUG ######################################################################
        # if video_id not in ["10294103914"]:
        #     continue
        ################################################################################

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
            
            # COLOR PALETTE : RED
            color_bgrs = []
            rgb = (255, 0, 0)
            for i in range(num_objects):
                color_bgrs.append(tuple(reversed(rgb)))

            for obj_idx, obj_id in enumerate(sorted(new_mask_dict.keys())):
                
                # PREPARE FOR VIDEO SAVING
                out_video_dir = os.path.join(args.output_dir, f"{video_id}_clip{split_idx:03d}", obj_id)
                if os.path.exists(out_video_dir) and (len(os.listdir(out_video_dir)) == args.num_segments):
                    continue
                os.makedirs(out_video_dir, exist_ok=True)
                
                # LOAD MASK
                rle_masklet = new_mask_dict[obj_id]
                assert len(rle_masklet) == len(frames)
                
                # SINGLE OBJECT : CROP FRAME (START, END)
                start_frame_idx, end_frame_idx = first_frame_dict[obj_id], last_frame_dict[obj_id]
                if (start_frame_idx is None) or (end_frame_idx is None):
                    assert start_frame_idx == end_frame_idx == None
                    shutil.rmtree(out_video_dir)
                    continue
                crop_frames = frames[start_frame_idx : end_frame_idx]
                crop_rle_masklet = rle_masklet[start_frame_idx : end_frame_idx]
                assert len(crop_rle_masklet) == len(crop_frames)

                # # SAVE CROPPED VIDEOS
                # crop_video_path = "temp_single.mp4"
                # writer = imageio.get_writer(crop_video_path, fps=30, codec='libx264')
                # for frame in crop_frames:
                #     img_path = os.path.join(vid_dir, frame)
                #     img = cv2.imread(img_path)
                #     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                #     writer.append_data(img_rgb)
                # writer.close()

                # # FRAMES THAT MLLM SEES
                # _, _, frame_indices = load_video(crop_video_path, num_segments=args.num_segments, return_indices=True)
                # os.remove(crop_video_path)

                # FRAMES THAT MLLM SEES
                frame_indices = get_frame_indices_only(len(crop_frames) - 1, num_segments=args.num_segments)
                
                # OBTAIN FRAMES THAT MLLM SEES
                sub_frames, sub_rle_masklet = [], []
                for frame_idx in frame_indices:
                    frame_idx = int(frame_idx)
                    sub_frames.append(crop_frames[frame_idx])
                    sub_rle_masklet.append(crop_rle_masklet[frame_idx])

                # DECODE MASKLET
                masklet = []
                for rle_mask in sub_rle_masklet:
                    if rle_mask is not None:
                        mask = mask_utils.decode(rle_mask).astype(np.uint8)
                    else:
                        mask = np.zeros((height, width)).astype(np.uint8)
                    masklet.append(mask*255)
                assert len(masklet) == len(sub_frames)
                
                # FILTER OBJECTS APPEAR LESS THAN THRESHOLD
                object_ratios = []
                for frame_idx, frame in enumerate(sub_frames):
                    mask = masklet[frame_idx]
                    object_area = int(mask.sum() / 255)
                    total_area = mask.size
                    object_ratio = object_area / total_area
                    object_ratios.append(object_ratio)
                zero_count = object_ratios.count(0.0)
                non_zero_count = args.num_segments - zero_count
                if len(new_mask_dict.keys()) > 3:
                    if (max(object_ratios) <= 0.01) or (non_zero_count <= 2):
                        shutil.rmtree(out_video_dir)
                        continue

                # MASK OVERLAY + FIRST FRAME + LABEL
                if args.vp_type == "":
                    first_frame = True
                    for frame_idx, frame in enumerate(sub_frames):
                        frame = cv2.imread(os.path.join(vid_dir, sub_frames[frame_idx]))
                        # RESIZE TALL AND HEIGHT MORE THAN 1000 PIXELS
                        if (height > width) and height >= 1000:
                            frame = cv2.resize(frame, (width // 2, height // 2), interpolation=cv2.INTER_AREA)
                            masklet[frame_idx] = cv2.resize(masklet[frame_idx], (width // 2, height // 2), interpolation=cv2.INTER_NEAREST)
                        if first_frame and (object_ratios[frame_idx] > 0.005):
                            processed_frame = mask_overlay_on_frame(frame, masklets=[masklet[frame_idx]], palette=[color_bgrs[obj_idx]], label=True)
                            first_frame = False
                        else:
                            processed_frame = mask_outline_on_frame(frame, masklets=[masklet[frame_idx]], palette=[color_bgrs[obj_idx]])
                        processed_frame = resize_to_even(processed_frame)
                        # SAVE IMAGE
                        save_path = os.path.join(out_video_dir, f"{frame_idx:06d}.jpg")
                        cv2.imwrite(save_path, processed_frame)

                # MASK OVERLAY + LABEL
                elif args.vp_type == "mask_overlay":
                    for frame_idx, frame in enumerate(sub_frames):
                        frame = cv2.imread(os.path.join(vid_dir, sub_frames[frame_idx]))
                        # RESIZE TALL AND HEIGHT MORE THAN 1000 PIXELS
                        if (height > width) and height >= 1000:
                            frame = cv2.resize(frame, (width // 2, height // 2), interpolation=cv2.INTER_AREA)
                            masklet[frame_idx] = cv2.resize(masklet[frame_idx], (width // 2, height // 2), interpolation=cv2.INTER_NEAREST)
                        processed_frame = mask_overlay_on_frame(frame, masklets=[masklet[frame_idx]], palette=[color_bgrs[obj_idx]], label=True)
                        processed_frame = resize_to_even(processed_frame)
                        # SAVE IMAGE
                        save_path = os.path.join(out_video_dir, f"{frame_idx:06d}.jpg")
                        cv2.imwrite(save_path, processed_frame)

    print("*** ERROR IN MASK DICT ***")
    print(mask_dict_error)
