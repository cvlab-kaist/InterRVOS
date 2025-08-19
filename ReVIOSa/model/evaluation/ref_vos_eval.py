import os
import torch
import json
import tqdm
import argparse
import numpy as np
from PIL import Image

import mmengine

import torch.distributed
import torch.utils.data
import concurrent.futures

from transformers import AutoModel, AutoTokenizer

from model.evaluation.dataset import RefVOSDataset
from model.evaluation.utils import _init_dist_pytorch, _init_dist_slurm, get_dist_info, get_rank, collect_results_cpu

from pycocotools import mask as cocomask


def uniform_sample_frames(frame_list, max_frames=10):
    n = len(frame_list)
    if n <= max_frames:
        return frame_list
    else:
        indices = np.linspace(0, n - 1, max_frames, dtype=int)
        return [frame_list[i] for i in indices]


def async_func(executor, func, **kwargs):
    future = executor.submit(func, **kwargs)
    return future


def mask_to_rle(mask):
    rle = []
    for m in mask:
        rle.append(cocomask.encode(np.asfortranarray(m.astype(np.uint8))))
        rle[-1]['counts'] = rle[-1]['counts'].decode()
    return rle


def mask_save(item, mask_prediction, work_dir):
    vid_id = item['video_id']
    exp_id = item['exp_id']
    save_path = os.path.join(work_dir, 'Annotations', vid_id, exp_id)
    mmengine.mkdir_or_exist(save_path)
    for id_m, mask in enumerate(mask_prediction):
        mask = Image.fromarray(mask.astype(np.float32) * 255).convert('L')
        file_name = item['frames'][id_m]
        save_file = os.path.join(save_path, file_name + ".png")
        mask.save(save_file)


DATASETS_INFO = {
    'INTERRVOS': {
        'data_root': 'datasets/InterRVOS/val/',
        'image_folder': 'datasets/InterRVOS/val/JPEGImages',
        'expression_file': 'datasets/InterRVOS/val/meta_expressions.json',
        'mask_file': 'datasets/InterRVOS/val/mask_dict.json',
    },
    'MEVIS': {
        'data_root': 'datasets/mevis/valid/',
        'image_folder': 'datasets/mevis/valid/JPEGImages',
        'expression_file': 'datasets/mevis/valid/meta_expressions.json',
        'mask_file': None,
    },
    'MEVIS_U': {
        'data_root': 'datasets/mevis/valid_u/',
        'image_folder': 'datasets/mevis/valid_u/JPEGImages',
        'expression_file': 'datasets/mevis/valid_u/meta_expressions.json',
        'mask_file': 'datasets/mevis/valid_u/mask_dict.json',
    },
    'REFYTVOS': {
        'data_root': 'datasets/ref_ytbvos/',
        'image_folder': 'datasets/ref_ytbvos/valid/JPEGImages/',
        'expression_file': 'datasets/ref_ytbvos/valid/meta_expressions.json',
        'mask_file': None,
    },
    'DAVIS': {
        'data_root': 'datasets/ref_davis/',
        'image_folder': 'datasets/ref_davis/valid/JPEGImages/',
        'expression_file': 'datasets/ref_davis/valid/meta_expressions.json',
        'mask_file': 'datasets/ref_davis/valid/mask_dict.json',
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description='RefVOS')
    parser.add_argument('model_path', help='hf model path.')
    parser.add_argument(
        '--dataset',
        choices=DATASETS_INFO.keys(),
        default='MEVIS',
        help='Specify a dataset')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--work_dir', type=str, default=None)
    parser.add_argument('--deepspeed', type=str, default=None) # dummy
    parser.add_argument('--n_pid', type=int, default=1, help='Total number of splits')
    parser.add_argument('--pid', type=int, default=0, help='Current split index')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


if __name__ == '__main__':
    args = parse_args()

    meta_dict = json.load(open(DATASETS_INFO[args.dataset]['expression_file'], "r"))["videos"]

    work_dir = args.work_dir
    if work_dir is None:
        work_dir = 'work_dirs/foobar'

    if args.launcher == 'none':
        rank = 0
        world_size = 1
    elif args.launcher == 'pytorch':
        _init_dist_pytorch('nccl')
        rank, world_size = get_dist_info()
    elif args.launcher == 'slurm':
        _init_dist_slurm('nccl')
        rank, world_size = get_dist_info()

    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
    ).eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    dataset_info = DATASETS_INFO[args.dataset]


    dataset = RefVOSDataset(
        image_folder=dataset_info['image_folder'],
        expression_file=dataset_info['expression_file'],
        mask_file=dataset_info['mask_file'],
        n_pid=args.n_pid,
        pid=args.pid,
    )

    sampler = torch.utils.data.DistributedSampler(
        dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=False,
        drop_last=False
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=1,
        num_workers=2,
        pin_memory=False,
        collate_fn=lambda x:x[0],
    )
    results = []
    executor = concurrent.futures.ThreadPoolExecutor()

    for item in tqdm.tqdm(dataloader):
        with torch.no_grad():
            result = model.predict_forward(
                video=item['images'],
                text=item['text_prompt'],
                tokenizer=tokenizer,
            )

        text_idx = 0
        text_prediction = result['prediction']
        
        print(text_prediction)
        if len(result['prediction_masks_act']) > 0:
            mask_prediction_act = result['prediction_masks_act'][text_idx]
        else:
            mask_prediction_act = np.zeros((item['length'], item['ori_height'], item['ori_width']), dtype=np.uint8)
        
        if len(result['prediction_masks_tar']) > 0:
            mask_prediction_tar = result['prediction_masks_tar'][text_idx]
        else:
            mask_prediction_tar = np.zeros((item['length'], item['ori_height'], item['ori_width']), dtype=np.uint8)

        if args.submit:
            assert args.dataset != "INTERRVOS"
            async_func(executor, mask_save, item=item, mask_prediction=mask_prediction_act, work_dir=work_dir)
            encoded_mask_act = None
            encoded_mask_tar = None
        else:
            encoded_mask_act = mask_to_rle(mask_prediction_act)
            encoded_mask_tar = mask_to_rle(mask_prediction_tar)

        result = {
            'index': item['index'],
            'video_id': item['video_id'],
            'exp_id': item['exp_id'],
            'text_prediction': text_prediction,
            'frames': item['frames'],
            'exp': item['text_prompt'],
            'prediction_masks_act': encoded_mask_act,
            'prediction_masks_tar': encoded_mask_tar,
        }
        results.append(result)


    executor.shutdown(wait=True)
    print(f'[Rank {rank}] : Finished.')
    
    if not args.submit:
        results = collect_results_cpu(results, len(dataset))
        if get_rank() == 0:
            final_results = {}
            for item in results:
                vid_id = item['video_id']
                exp_id = item['exp_id']
                if vid_id not in final_results:
                    final_results[vid_id] = {}
                assert exp_id not in final_results[vid_id]
                final_results[vid_id][exp_id] = item
            os.makedirs(work_dir, exist_ok=True)
            json.dump(final_results, open(f'{work_dir}/results_pid_{args.pid}.json', 'w'), indent=4)

    if rank == 0:
        print('Done')
