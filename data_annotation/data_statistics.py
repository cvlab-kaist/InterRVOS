import os
import json
import glob

raw_json = "datasets/VidOR/FINAL/final_filtered.json"
with open(raw_json, "r") as f:
    raw_dict = json.load(f)
data_json = "datasets/VidOR/FINAL/meta_expressions.json"
with open(data_json, "r") as f:
    meta_dict = json.load(f)["videos"]

num_videos = len(meta_dict.keys())
num_exps = []
for video_id, infos in meta_dict.items():
    num_exps.append(len(infos['expressions']))

num_objects = []
for video_id, infos in raw_dict.items():
    num_objects.append(len(infos['gpt_single_object']))

print(f"Number of videos: {num_videos}")
print(f"Number of expressions: {sum(num_exps)}")
print(f"Number of expressions per video: {sum(num_exps) / num_videos}\n")

print(f"Number of objects: {sum(num_objects)}")
print(f"Number of objects per video: {sum(num_objects) / num_videos}\n")

n_single, n_multi, n_bi, n_uni = [], [], [], []
for video_id, infos in meta_dict.items():
    for exp_id, exp_info in infos["expressions"].items():
        if exp_info["caption_type"] == "single":
            n_single.append(1)
        elif exp_info["caption_type"] == "multi":
            n_multi.append(1)
        elif exp_info["caption_type"] == "unidirectional":
            n_uni.append(1)
        elif exp_info["caption_type"] == "bidirectional":
            n_bi.append(1)
        else:
            raise ValueError(f"Unknown caption type: {exp_info['caption_type']}")

print(f"Number of single object expressions: {sum(n_single)}")
print(f"Number of multi object expressions: {sum(n_multi)}")
print(f"Number of bidirectional interactions: {sum(n_bi)}")
print(f"Number of unidirectional interactions: {sum(n_uni)}\n")

print(f"Number of single object expressions per video: {sum(n_single) / num_videos}")
print(f"Number of multi object expressions per video: {sum(n_multi) / num_videos}")
print(f"Number of bidirectional interactions per video: {sum(n_bi) / num_videos}")
print(f"Number of unidirectional interactions per video: {sum(n_uni) / num_videos}\n")

# Ratio of each caption type
print(f"Ratio of single object expressions: {sum(n_single) / sum(num_exps)}")
print(f"Ratio of multi object expressions: {sum(n_multi) / sum(num_exps)}")
print(f"Ratio of bidirectional interactions: {sum(n_bi) / sum(num_exps)}")
print(f"Ratio of unidirectional interactions: {sum(n_uni) / sum(num_exps)}\n")
