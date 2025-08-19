###########################################################################
# Created by: NTU
# Email: heshuting555@gmail.com
# Copyright (c) 2023
###########################################################################
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
import os
import os.path as osp
import time
import argparse
import json
import numpy as np
from pycocotools import mask as cocomask

from multiprocessing import Value
from tqdm import tqdm

from third_parts.revos.utils.metrics import db_eval_iou, db_eval_boundary
import multiprocessing as mp

NUM_WOEKERS = 128


def eval_queue(q, rank, out_dict, progress_counter):
    while not q.empty():
        # print(q.qsize())
        vid_name, exp = q.get()

        vid = exp_dict[vid_name]

        exp_name = f'{vid_name}_{exp}'
        if vid_name not in mask_pred_dict: continue
        pred = mask_pred_dict[vid_name][exp]

        h, w = pred['prediction_masks_act'][0]['size']
        vid_len = len(vid['frames'])
        gt_masks = np.zeros((vid_len, h, w), dtype=np.uint8)
        pred_masks = np.zeros((vid_len, h, w), dtype=np.uint8)

        anno_ids = vid['expressions'][exp]['anno_id']

        for frame_idx, frame_name in enumerate(vid['frames']):
            for anno_id in anno_ids:
                mask_rle = mask_dict[str(anno_id)][frame_idx]
                if mask_rle:
                    gt_masks[frame_idx] += cocomask.decode(mask_rle)

            pred_mask = cocomask.decode(pred['prediction_masks_act'][frame_idx])
            pred_masks[frame_idx] += pred_mask

        j = db_eval_iou(gt_masks, pred_masks).mean()
        f = db_eval_boundary(gt_masks, pred_masks).mean()
        out_dict[exp_name] = [j, f]

        with progress_counter.get_lock():
            progress_counter.value += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", type=str, default="results.json")
    parser.add_argument("--data_dir", type=str, default="datasets/interrvos/val")
    parser.add_argument("--save_path", type=str, default="eval.json")
    args = parser.parse_args()
    queue = mp.Queue()

    exp_path = os.path.join(args.data_dir, "new_meta_expressions.json")
    mask_path = os.path.join(args.data_dir, "new_mask_dict.json")
    
    exp_dict = json.load(open(exp_path))['videos']
    mask_dict = json.load(open(mask_path))

    shared_exp_dict = mp.Manager().dict(exp_dict)
    shared_mask_dict = mp.Manager().dict(mask_dict)
    output_dict = mp.Manager().dict()

    mask_pred = json.load(open(args.pred_path))
    mask_pred_dict  = mp.Manager().dict(mask_pred)

    for vid_name in exp_dict:
        vid = exp_dict[vid_name]
        for exp in vid['expressions']:
            queue.put([vid_name, exp])

    # tqdm
    total_tasks = queue.qsize()
    progress_counter = Value('i', 0)

    start_time = time.time()
    if NUM_WOEKERS > 1:
        processes = []
        for rank in range(NUM_WOEKERS):
            p = mp.Process(target=eval_queue, args=(queue, rank, output_dict, progress_counter))
            p.start()
            processes.append(p)

        # tqdm
        with tqdm(total=total_tasks) as pbar:
            prev = 0
            while any(p.is_alive() for p in processes):
                with progress_counter.get_lock():
                    delta = progress_counter.value - prev
                    prev = progress_counter.value
                pbar.update(delta)
                time.sleep(0.1)

        for p in processes:
            p.join()
    else:
        eval_queue(queue, 0, output_dict, progress_counter)

    j = [output_dict[x][0] for x in output_dict]
    f = [output_dict[x][1] for x in output_dict]
    
    output_dict = dict(output_dict)
    output_dict["final"] = {
        "J": round(100 * float(np.mean(j)), 2),
        "F": round(100 * float(np.mean(f)), 2),
        "J&F": round(100 * float((np.mean(j) + np.mean(f)) / 2), 2),
    }
    
    with open(args.save_path, 'w') as file:
        json.dump(output_dict, file, indent=4)

    end_time = time.time()
    total_time = end_time - start_time
    print("time: %.4f s" % (total_time))

