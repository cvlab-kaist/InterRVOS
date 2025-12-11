import os
import glob
import json
from tqdm import tqdm


final_json = "datasets/VidOR/FINAL/final_filtered.json"
with open(final_json, "r") as f:
    data_generated = json.load(f)

mask_root = "datasets/VidOR/MaskAnnotationsV4"
image_root = "datasets/VidOR/JPEGImages_clips"

# new annotations
referring_json_path = "datasets/VidOR/FINAL/meta_expressions.json"
mask_json_path = "datasets/VidOR/FINAL/mask_dict.json"

meta_dict, mask_dict = {}, {}
anno_dict, anno_max = {}, 0
vid_id_max = 0

meta_dict["videos"] = {}

# PROCESSING META EXPRESSIONS & MASK DICT
for video_id, generated_captions in tqdm(data_generated.items()):

    ########## SORRY MY MISTAKE.... ##########
    if "4178206790" in video_id:
        continue

    # CHECK IF ALL KEYS ARE PRESENT
    if not all(key in generated_captions for key in ["gpt_single_object", "gpt_interaction", "llama_single_object", "llama_multi_object", "llama_interaction"]):
        continue
    # CHECK IF THE GENERATED CAPTIONS ARE NOT NONE
    if not all(sub_process_captions != None for sub_process_captions in generated_captions.values()):
        continue
    # CHECK IF NUMBER OF GPT SINLGE OBJECT CAPTIONS AND LLAMA SINGLE OBJECT CAPTIONS ARE SAME
    if len(generated_captions["gpt_single_object"]) != len(generated_captions["llama_single_object"]):
        continue

    # FRAMES
    vid_dir = os.path.join(image_root, video_id.split('_')[0], f"clip_{video_id.split('_')[1].replace('clip', '')}")
    frames = sorted(os.listdir(vid_dir))[:500]
    
    # LOAD MASK FOR EACH OBJECT FOR EACH VIDEO CLIP
    mask_json = os.path.join(mask_root, f"{video_id.split('_')[0]}_{int(video_id.split('_')[1].replace('clip', ''))}.json")
    with open(mask_json, "r") as f:
        mask_data = json.load(f)

    meta_dict["videos"][video_id] = {}    
    meta_dict["videos"][video_id]["expressions"] = {}
    meta_dict["videos"][video_id]["vid_id"] = vid_id_max
    vid_id_max += 1
    
    meta_dict["videos"][video_id]["frames"] = [frames[i].replace(".jpg", "") for i in range(len(frames))]
    
    anno_dict[video_id] = {}
    exp_id = 0

    # VALID OBJECTS IN A VIDEO (GPT SINGLE OBJECT)
    valid_obj_ids = list(generated_captions["gpt_single_object"].keys())

    # SINGLE OBJECT
    for obj_id, captions in generated_captions["llama_single_object"].items():
        # CHECK IF THE CAPTION IS VALID
        if obj_id not in valid_obj_ids:
            continue
        
        anno_dict[video_id][obj_id] = anno_max
        anno_max += 1
        
        for caption_type, caption in captions.items():
            meta_dict["videos"][video_id]["expressions"][str(exp_id)] = {}
            meta_dict["videos"][video_id]["expressions"][str(exp_id)]["exp"] = caption
            meta_dict["videos"][video_id]["expressions"][str(exp_id)]["obj_id"] = [int(obj_id)]
            meta_dict["videos"][video_id]["expressions"][str(exp_id)]["anno_id"] = [anno_dict[video_id][obj_id]]
            meta_dict["videos"][video_id]["expressions"][str(exp_id)]["caption_type"] = "single"
            exp_id += 1

    # MULTI OBJECT
    if generated_captions["llama_multi_object"]["merged"] == "YES":
        obj_ids, anno_ids = [], []
        for obj_id in generated_captions["llama_multi_object"]['merged_objects']:
            if isinstance(obj_id, int):
                obj_id = str(obj_id)
            elif ("[" in obj_id) or ("]" in obj_id):
                obj_id = obj_id.replace("[", "").replace("]", "")
            obj_ids.append(int(obj_id))
            anno_ids.append(anno_dict[video_id][obj_id])
        
        # CHECK IF THE CAPTION IS VALID
        if obj_ids != []:
            meta_dict["videos"][video_id]["expressions"][str(exp_id)] = {}
            meta_dict["videos"][video_id]["expressions"][str(exp_id)]['exp'] = generated_captions["llama_multi_object"]["merged_caption"]
            meta_dict["videos"][video_id]["expressions"][str(exp_id)]["obj_id"] = obj_ids
            meta_dict["videos"][video_id]["expressions"][str(exp_id)]["anno_id"] = anno_ids
            if len(obj_ids) > 1:
                meta_dict["videos"][video_id]["expressions"][str(exp_id)]["caption_type"] = "multi"
            else:
                meta_dict["videos"][video_id]["expressions"][str(exp_id)]["caption_type"] = "single"
            exp_id += 1

    # INTERACTION
    if (generated_captions["gpt_interaction"]["interaction"] == "YES") and (generated_captions["llama_interaction"][0]["obj_id"] != []):
        for interaction_idx, interaction in enumerate(generated_captions["llama_interaction"]):
            valid_interaction = True
            obj_ids, anno_ids = [], []
            for obj_id in interaction["obj_id"]:
                if isinstance(obj_id, int):
                    obj_id = str(obj_id)
                elif ("[" in obj_id) or ("]" in obj_id):
                    obj_id = obj_id.replace("[", "").replace("]", "")
                if obj_id not in anno_dict[video_id]:
                    valid_interaction = False
                    break
                obj_ids.append(int(obj_id))
                anno_ids.append(anno_dict[video_id][obj_id])
            
            ### DEBUG
            if video_id == "13936526931_clip000": break
            ### DEBUG
            
            if not valid_interaction:
                print(f"SKIP INTERACTION | VIDEO {video_id}")
                break
            
            # CHECK IF THE CAPTION IS VALID
            if valid_interaction:
                # CLASS-LEVEL
                meta_dict["videos"][video_id]["expressions"][str(exp_id)] = {}
                meta_dict["videos"][video_id]["expressions"][str(exp_id)]['exp'] = interaction['class_level']
                meta_dict["videos"][video_id]["expressions"][str(exp_id)]["obj_id"] = obj_ids
                meta_dict["videos"][video_id]["expressions"][str(exp_id)]["anno_id"] = anno_ids
                if interaction["interaction_type"] == "bidirectional":
                    meta_dict["videos"][video_id]["expressions"][str(exp_id)]["caption_type"] = "bidirectional"
                elif interaction["interaction_type"] == "unidirectional":
                    meta_dict["videos"][video_id]["expressions"][str(exp_id)]["caption_type"] = "unidirectional"
                exp_id += 1

                # APPEARANCE-LEVEL
                meta_dict["videos"][video_id]["expressions"][str(exp_id)] = {}
                meta_dict["videos"][video_id]["expressions"][str(exp_id)]['exp'] = interaction['appearance_level']
                obj_ids, anno_ids = [], []
                for obj_id in interaction["obj_id"]:
                    if isinstance(obj_id, int):
                        obj_id = str(obj_id)
                    elif ("[" in obj_id) or ("]" in obj_id):
                        obj_id = obj_id.replace("[", "").replace("]", "")
                    if obj_id not in anno_dict[video_id]:
                        continue
                    obj_ids.append(int(obj_id))
                    anno_ids.append(anno_dict[video_id][obj_id])
                meta_dict["videos"][video_id]["expressions"][str(exp_id)]["obj_id"] = obj_ids
                meta_dict["videos"][video_id]["expressions"][str(exp_id)]["anno_id"] = anno_ids
                if interaction["interaction_type"] == "bidirectional":
                    meta_dict["videos"][video_id]["expressions"][str(exp_id)]["caption_type"] = "bidirectional"
                elif interaction["interaction_type"] == "unidirectional":
                    meta_dict["videos"][video_id]["expressions"][str(exp_id)]["caption_type"] = "unidirectional"
                exp_id += 1

    # SAVE MASKS
    for obj_id, masklet in mask_data.items():
        if isinstance(obj_id, int):
            obj_id = str(obj_id)
        elif ("[" in obj_id) or ("]" in obj_id):
            obj_id = obj_id.replace("[", "").replace("]", "")
        if obj_id not in anno_dict[video_id]:
            continue
        mask_dict[str(anno_dict[video_id][obj_id])] = mask_dict.get(str(anno_dict[video_id][obj_id]), [])
        mask_dict[str(anno_dict[video_id][obj_id])] = masklet["mask"][:500]

# SAVE JSON FILES
print("SAVING META DICT...")
with open(referring_json_path, "w") as f:
    json.dump(meta_dict, f, indent=4)
print("SAVING MASK DICT...")
with open(mask_json_path, "w") as f:
    json.dump(mask_dict, f, indent=4)
