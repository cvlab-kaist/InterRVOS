import os
import json
import argparse


def merge_json_files(result_dir, save_path):
    
    json_files = [f for f in os.listdir(result_dir) if f.endswith('.json')]

    merged_data = {}
    for json_file in json_files:
        file_path = os.path.join(result_dir, json_file)

        with open(file_path, "r") as f:
            data = json.load(f)

        for video_id, info in data.items():
            merged_data[video_id] = merged_data.get(video_id, {})
            for obj_id, obj_info in info.items():
                assert obj_id not in merged_data[video_id]
                merged_data[video_id][obj_id] = obj_info
    
    with open(save_path, "w") as f:
        json.dump(merged_data, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, default="ReVIOSa/EVAL/reviosa_1b_interrvos/interrvos")
    parser.add_argument("--save_path", type=str, default="ReVIOSa/EVAL/reviosa_1b_interrvos/interrvos/results.json")
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        merge_json_files(args.result_dir, args.save_path)
