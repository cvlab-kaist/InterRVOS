import os
import cv2
import glob
import time
import re
import math
import shutil
import json
import argparse
from PIL import Image
from tqdm import tqdm

from openai import OpenAI
import base64

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from utils.vid2img import extract_frames
from utils.prompts import make_prompt

client = OpenAI()

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def check_single_caption_valid(caption):
    # 1. GPT SAYS SORRY
    if ("I'm" in caption) or ("cannot" in caption) or ("can't" in caption) or ("unable" in caption) or ("sorry" in caption):
        return True
    # 2. CAPTION IS NOT PROPERLY FORMATTED
    elif len(caption.split(". ")) < 2:
        return True
    else:
        return False


def load_json(json_path):
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return_dict = json.load(f)
    else:
        return_dict = {}
    
    return return_dict


def clean_and_parse_generated_text(generated_text):
    # Extract the first JSON block using regex
    match = re.search(r'\{[\s\S]*\}', generated_text)
    
    if not match:
        print("No JSON block found.")
        return None

    json_block = match.group(0)

    # Replace Python None with JSON null, just in case
    json_block = json_block.replace("None", "null")

    try:
        parsed = json.loads(json_block)
    except json.JSONDecodeError as e:
        print("JSON decode error:", e)
        print("Extracted JSON block:\n", json_block)
        parsed = None
    
    return parsed


def split_into_app_and_mot(caption):
    try:
        appearance_caption, motion_caption = caption.split(". ", 1)
    except:
        if '\"' in caption:
            caption = caption.replace("\"", "")
            appearance_caption, motion_caption = caption.split(". ", 1)
    
    return appearance_caption, motion_caption


def gpt_single_object_caption(video_dir, class_anno_dir):
    
    video_paths = sorted(glob.glob(os.path.join(video_dir, "*")))
    video_id = os.path.basename(video_dir)

    class_anno_path = glob.glob(os.path.join(class_anno_dir, "**", f"{video_id.split('_')[0]}.json"))[0]
    with open(class_anno_path, "r") as f:
        obj_class_list = json.load(f)["subject/objects"]

    generated_dict = {}
    for video_path in video_paths:
        
        obj_id = os.path.basename(video_path)
        obj_info = obj_class_list[int(obj_id)]
        assert obj_info['tid'] == int(obj_id)
        obj_class = obj_info["category"]

        generated_dict[obj_id] = {}
        
        # PROMPT
        prompt = make_prompt(
            stage="single_object_caption",
            obj_class=obj_class,
            )

        # IMAGE PATHS
        image_paths = sorted(glob.glob(os.path.join(video_path, "*")))

        # Build the message content
        content = [{"type": "text", "text": prompt}]
        for idx, path in enumerate(image_paths):
            frame_number = f"Frame {idx:04d}"
            # Add frame label as text
            content.append({
                "type": "text",
                "text": f"[{frame_number}]"
            })
            # Add the corresponding image
            base64_image = encode_image(path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })

        # CHAT
        try:
            chat_completion = client.chat.completions.create(
                model="gpt-4o",
                max_tokens=1000,
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ]
            )
            content_text = chat_completion.choices[0].message.content
            
        except Exception as e:
            print(e)
            error_message = str(e)

            match = re.search(r"try again in (\d+(?:\.\d+)?)(ms|s)", error_message)
            if match:
                value, unit = match.groups()
                value = float(value)

                wait_time = math.ceil(value / 1000) if unit == "ms" else math.ceil(value)
                print(f"Rate limit hit. Waiting for {wait_time} seconds...")
                time.sleep(wait_time)

                chat_completion = client.chat.completions.create(
                    model="gpt-4o",
                    max_tokens=1000,
                    messages=[
                        {
                            "role": "user",
                            "content": content
                        }
                    ]
                )
                content_text = chat_completion.choices[0].message.content
        
        generated_dict[obj_id] = content_text

    return generated_dict


def gpt_single_object_caption_again(video_dir, class_anno_dir, obj_id):
    
    video_paths = sorted(glob.glob(os.path.join(video_dir, "*")))
    video_id = os.path.basename(video_dir)

    class_anno_path = glob.glob(os.path.join(class_anno_dir, "**", f"{video_id.split('_')[0]}.json"))[0]
    with open(class_anno_path, "r") as f:
        obj_class_list = json.load(f)["subject/objects"]
    
    for video_path in video_paths:
        if obj_id == os.path.basename(video_path):
            video_path_again = video_path

    assert os.path.basename(video_path_again) == obj_id

    obj_info = obj_class_list[int(obj_id)]
    assert obj_info['tid'] == int(obj_id)
    
    obj_class = obj_info["category"]
        
    # PROMPT
    prompt = make_prompt(
        stage="single_object_caption",
        obj_class=obj_class,
        )

    # IMAGE PATHS
    image_paths = sorted(glob.glob(os.path.join(video_path_again, "*")))

    # Build the message content
    content = [{"type": "text", "text": prompt}]
    for idx, path in enumerate(image_paths):
        frame_number = f"Frame {idx:04d}"
        # Add frame label as text
        content.append({
            "type": "text",
            "text": f"[{frame_number}]"
        })
        # Add the corresponding image
        base64_image = encode_image(path)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })

    # CHAT
    chat_completion = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=1000,
        messages=[
            {
                "role": "user",
                "content": content
            }
        ]
    )
    content_text = chat_completion.choices[0].message.content

    return content_text


def gpt_interaction_object_caption(video_path, mask_anno_dir):

    video_id = os.path.basename(video_path)
    
    # VALID OBJECT IDS
    with open(f"{os.path.dirname(video_path)}/valid_obj_ids.json") as f:
        valid_obj_ids_dict = json.load(f)[video_id]

    # OBJECT CATEGOIRES
    obj_categories = {}
    mask_anno_path = os.path.join(mask_anno_dir, f'{video_id.split("_")[0]}_{int(video_id.split("_")[1].replace("clip", ""))}.json')
    with open(mask_anno_path, "r") as f:
        mask_anno_dict = json.load(f)
    for obj_idx, infos in mask_anno_dict.items():
        if obj_idx in valid_obj_ids_dict:
            obj_categories[f"[{obj_idx}]"] = infos["category"]
    
    valid_obj_ids = []
    for obj_id in valid_obj_ids_dict:
        valid_obj_ids.append(f"[{obj_id}]")
    
    # PROMPT
    prompt = make_prompt(
        stage="object_interaction",
        obj_categories=repr(obj_categories),
        valid_obj_ids=", ".join(valid_obj_ids),
        )

    # IMAGE PATHS
    image_paths = sorted(glob.glob(os.path.join(video_path, "*")))

    # Build the message content
    content = [{"type": "text", "text": prompt}]
    for idx, path in enumerate(image_paths):
        frame_number = f"Frame {idx:04d}"
        # Add frame label as text
        content.append({
            "type": "text",
            "text": f"[{frame_number}]"
        })
        # Add the corresponding image
        base64_image = encode_image(path)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })

    # CHAT
    try:
        chat_completion = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1000,
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ]
        )
        content_text = chat_completion.choices[0].message.content

    except Exception as e:
        print(e)
        error_message = str(e)
        
        match = re.search(r"try again in (\d+(?:\.\d+)?)(ms|s)", error_message)
        if match:
            value, unit = match.groups()
            value = float(value)

            wait_time = math.ceil(value / 1000) if unit == "ms" else math.ceil(value)
            print(f"Rate limit hit. Waiting for {wait_time} seconds...")
            time.sleep(wait_time)
            
            chat_completion = client.chat.completions.create(
                model="gpt-4o",
                max_tokens=1000,
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ]
            )
            content_text = chat_completion.choices[0].message.content
    
    generated_dict = clean_and_parse_generated_text(content_text)
    
    return generated_dict


def llama_single_object_caption(gpt_single_generated_dict):
    
    generated_dict = {}

    for obj_id, caption in gpt_single_generated_dict.items():
        
        gpt_appearance_caption, gpt_motion_caption = split_into_app_and_mot(caption)
        
        # PROMPT
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an assistant that generates **referring captions** for a single object in a video. "
                    "You will be given two descriptions of the object:\n"
                    "- An **appearance** description (what it looks like)\n"
                    "- A **motion** description (how it moves or changes position)\n\n"
                    "Your task is to convert these descriptions into **natural referring expressions**, while preserving as much information as possible.\n\n"
                    "**Generate three outputs:**\n"
                    "1. A caption that combines both appearance and motion (key: 'all')\n"
                    "2. A caption that uses only the motion (key: 'motion')\n"
                    "3. A caption that uses only the appearance (key: 'appearance')\n\n"
                    "**IMPORTANT RULES:**\n"
                    "- Rewrite each caption as a **referring expression**, not a full sentence.\n"
                    "- Use singular form only. Never use plural expressions like 'they' or 'their'. Assume the object is a single entity.\n"
                    "- Do not use the word ‘figure’. Use an alternative. Especially for the ‘motion’ description, use terms like ‘object’ or others that do not imply appearance.\n"
                    "- **Do not omit details** from the input descriptions. Keep the meaning and key attributes intact.\n"
                    "- Rephrase **only as needed** to make the output sound like a natural referring phrase.\n"
                    "- Do NOT add new information or hallucinate.\n"
                    "- Avoid phrases like 'The object is' or 'This is'.\n\n"
                    "**Output must be in the following strict JSON format:**\n"
                    "{\n"
                    '  "all": "<caption combining appearance and motion>",\n'
                    '  "motion": "<caption using only motion>",\n'
                    '  "appearance": "<caption using only appearance>"\n'
                    "}"
                )
            },
            {
                "role": "user",
                "content": (
                    f"appearance_caption: {gpt_appearance_caption}\n"
                    f"motion_caption: {gpt_motion_caption}\n\n"
                    "Please generate the referring captions in the specified JSON format, following the rules above."
                )
            }
        ]
        prompts = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        
        # RUN LLM
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        generated_text = outputs[0].outputs[0].text
        generated_dict[obj_id] = clean_and_parse_generated_text(generated_text)

    return generated_dict


def llama_multi_object_caption(gpt_single_generated_dict):

    video_objs_caption_dict = {}
    for obj_id, caption in gpt_single_generated_dict.items():
        gpt_appearance_caption, gpt_motion_caption = split_into_app_and_mot(caption)
        video_objs_caption_dict[obj_id] = {
            "appearance_caption" : gpt_appearance_caption,
            "motion_caption" : gpt_motion_caption,
        }

    # PROMPT
    messages = [
        {
            "role": "system",
            "content": (
                "You are an assistant that analyzes multiple objects in a video based on their motion captions. "
                "Your task is to determine whether any objects can be grouped together into a single referring caption, based on whether they:\n\n"
                "1. Belong to the **same object class** (e.g., person, hand, cup, phone)\n"
                "2. Share **semantically similar motion behaviors**\n"
                "3. Are **describing the same primary object** (not just interacting with the same object)\n\n"
                "**IMPORTANT RULES:**\n"
                "- For each object, only consider the **main object being described** in its motion caption. "
                "Do NOT merge objects that describe **different entities**, even if similar objects are mentioned in the background.\n"
                "- For example, 'A hand holding a phone' and 'A phone moving near the face' describe different main subjects (hand vs. phone) and should NOT be merged.\n"
                "- If the motion captions indicate that the objects are **stationary** or show **no meaningful movement**, then do NOT merge them. "
                "Only merge objects that share clear and active motion behaviors (e.g., crawling, lowering, walking, waving, spinning, moving around, sitting at a couch, watching TV).\n\n"
                "**Output Format (JSON only):**\n"
                "- 'merged': 'YES' or 'NO'\n"
                "- 'merged_objects': List of object IDs that were merged (or null if no merge)\n"
                "- 'merged_caption': Referring caption describing the shared motion (or null if no merge)\n\n"
                "**Stylistic Rules for merged_caption:**\n"
                "- Use explicit object class (e.g., 'the people', 'the cups') — do not use pronouns like 'they'.\n"
                "- Write a **referring-style phrase**, not an explanatory sentence. Example: 'People walking side by side', not 'The people are walking...'\n"
                "- Your output must be valid JSON. No extra text or commentary."
            )
        },
        {
            "role": "user",
            "content": (
                f"obj_captions: {video_objs_caption_dict}\n\n"
                "Please determine if any objects can be merged based on object class and motion similarity, and return the result in the specified JSON format."
            )
        }
    ]
    prompts = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        
    # RUN LLM
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    generated_text = outputs[0].outputs[0].text
    generated_dict = clean_and_parse_generated_text(generated_text)

    return generated_dict


def llama_interaction_caption(gpt_single_generated_dict, gpt_interaction_generated_dict):

    generated_list = []

    video_objs_caption_dict = {}
    for obj_id, caption in gpt_single_generated_dict.items():
        gpt_appearance_caption, gpt_motion_caption = split_into_app_and_mot(caption)
        video_objs_caption_dict[f"[{obj_id}]"] = gpt_appearance_caption

    interaction_info = gpt_interaction_generated_dict
    if interaction_info is None:
        return None
    if interaction_info['interaction'] == "YES":
        for interaction_idx, interaction in enumerate(interaction_info["interactions"]):
            obj_captions = {}
            not_valid_interaction = False
            for obj in interaction["object_pair"]:
                if obj in video_objs_caption_dict:
                    obj_captions[obj] = video_objs_caption_dict[obj]
                else:
                    print(f"Skipping interaction {interaction_idx} for object {obj} not in video_objs_caption_dict")
                    not_valid_interaction = True
            if not_valid_interaction:
                continue
            interaction_type = interaction["type"]
            for interaction_description in interaction["descriptions"]:
                if interaction_type == "bidirectional":
                    object_ids = re.findall(r"\[\d+\]", interaction_description)
                    # PROMPT
                    messages = [
                        {
                            "role": "system",
                            "content": (
                                "You are an assistant that generates **referring captions** describing interactions between objects in a video.\n\n"
                                "Input:\n"
                                "- 'obj_captions': a dictionary of object IDs mapped to their appearance descriptions\n"
                                "- 'interaction_description': a natural language sentence involving object IDs (e.g., 'Object [0] and object [1] are sparring.')\n\n"
                                "Your task is to generate two types of referring captions by replacing the object references in the interaction_description "
                                "with natural expressions that identify them:\n"
                                "1. **class_level**: Use high-level object class names only (e.g., 'person', 'child')\n"
                                "2. **appearance_level**: Use short, distinguishing appearance descriptions (not full captions, just enough to tell them apart)\n\n"
                                "**Output Format:**\n"
                                "- Return a dictionary in JSON format with the following two keys:\n"
                                "    - 'class_level'\n"
                                "    - 'appearance_level'\n\n"
                                "**Stylistic Rules:**\n"
                                "- Referring captions must be concise and natural phrases (not explanatory sentences)\n."
                                "- Do NOT write full explanatory sentences like 'The A is doing B with the C'\n."
                                "Instead, write expressions like 'A doing B with C' or 'The one in red jacket sparring with the one in white shirt'\n."
                                "- You may omit verbs like 'is' or 'are' to keep the sentence minimal and referential in style\n."
                                "- Do NOT use pronouns like 'they' or 'their'.\n"
                                "- Do NOT write full sentences like 'The people are...'. Instead, write: 'People sparring with each other'.\n"
                                "- If both objects belong to the same class, you may use a plural collective form like 'People', 'Children', etc.\n"
                                "- The appearance-level caption should reflect just enough visual detail from obj_captions to distinguish the two objects naturally."
                            )
                        },
                        {
                            "role": "user",
                            "content": (
                                f"obj_captions: {obj_captions}\n"
                                f"interaction_description: {interaction_description}\n\n"
                                "Please return your response as a JSON dictionary containing the referring captions."
                            )
                        }
                    ]
                elif interaction_type == "unidirectional":
                    # SUBJECT/OBJECT ID
                    object_ids = re.findall(r"\[\d+\]", interaction_description)
                    if len(object_ids) == 2:
                        subject_id, object_id = object_ids
                    else:
                        subject_id, object_id = "Unknown, assume the subject ID", "Unknown, assume the object ID"
                    
                    # PROMPT
                    messages = [
                        {
                            "role": "system",
                            "content": (
                                "You are an assistant that generates **referring captions** describing interactions between objects in a video.\n\n"
                                "Input:\n"
                                "- 'obj_captions': a dictionary of object IDs mapped to their appearance descriptions\n"
                                "- 'interaction_description': a natural language sentence involving object IDs (e.g., 'Object [0] is hugging object [1]')\n"
                                "- 'subject_id': the ID of the object performing the action\n"
                                "- 'object_id': the ID of the object receiving the action\n\n"
                                "Your task is to generate two types of referring captions:\n"
                                "1. **class_level**: Use object class names only (e.g., 'person', 'cup', 'bear')\n"
                                "2. **appearance_level**: Use short, distinguishing appearance descriptions (not the full description — just enough to distinguish the object)\n\n"
                                "**Output Format:**\n"
                                "- Return a JSON dictionary with keys:\n"
                                "  - 'class_level'\n"
                                "  - 'appearance_level'\n\n"
                                "**Important Rules:**\n"
                                "- Carefully reflect the subject (agent) and object (recipient) roles as provided in 'subject_id' and 'object_id'.\n"
                                "- Do NOT follow the order in the sentence — follow the subject-object mapping explicitly.\n"
                                "- The referring captions must be short, descriptive, and in the form of **natural referring phrases** — not full explanatory sentences.\n"
                                "- Avoid structures like 'The A is doing B to the C'. Instead, use expressions like:\n"
                                "    - 'Parrot watching at person'\n"
                                "    - 'Person feeding a rabbit'\n"
                                "- Do NOT use pronouns like 'they' or 'their'.\n"
                                "- The appearance-level caption should reflect just enough visual detail from obj_captions to distinguish the two objects naturally."
                            )
                        },
                        {
                            "role": "user",
                            "content": (
                                f"obj_captions: {obj_captions}\n"
                                f"interaction_description: {interaction_description}\n"
                                f"subject_id: {subject_id}\n"
                                f"object_id: {object_id}\n\n"
                                "Please return your response as a JSON dictionary containing the referring captions.\n"
                                "Do **not** include any other description, explanation, or formatting — just the JSON dictionary."
                            )
                        }
                    ]
                prompts = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                
                # RUN LLM
                outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
                generated_text = outputs[0].outputs[0].text
                
                # SAVE GENERATED CAPTIONS
                content = clean_and_parse_generated_text(generated_text)

                if content is None:
                    ### DEBUG ########################################
                    print(f"Error in cleaning {generated_text}")
                    generated_dict = {
                        "obj_id" : object_ids,
                        "interaction_type" : interaction_type,
                        "class_level" : "Not generated properly",
                        "appearance_level" : "Not generated properly",
                    }
                    ##################################################
                else:
                    if interaction_type == "bidirectional":
                        generated_dict = {
                            "obj_id" : object_ids,
                            "interaction_type" : interaction_type,
                            "class_level" : content["class_level"],
                            "appearance_level" : content["appearance_level"],
                        }
                    elif interaction_type == "unidirectional":
                        generated_dict = {
                            "obj_id" : [object_ids[0]],
                            "interaction_type" : interaction_type,
                            "class_level" : content["class_level"],
                            "appearance_level" : content["appearance_level"],
                        }
                
                generated_list.append(generated_dict)
        
        if generated_list == []:
            generated_list = [
                {
                    "obj_id" : [],
                    "interaction_type" : None,
                    "class_level" : None,
                    "appearance_level" : None,
                }
            ] 
    
    else:
        generated_list = [
            {
                "obj_id" : [],
                "interaction_type" : None,
                "class_level" : None,
                "appearance_level" : None,
            }
        ]
    
    return generated_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_dir', type=str, default="processed")
    parser.add_argument('--stage', type=str, default="")
    parser.add_argument('--mask_anno_dir', type=str, default="datasets/VidOR/MaskAnnotations/train")
    parser.add_argument('--class_anno_dir', type=str, default="datasets/VidOR/Annotations/train")
    parser.add_argument('--output_dir', type=str, default="dir/to/save/results")
    parser.add_argument('--num_samples', type=int, default=0)
    parser.add_argument('--num_gpus', type=int, default=4)
    args = parser.parse_args()

    # SAVE PATHS
    gpt_single_json = os.path.join(args.output_dir, "gpt_single.json")
    gpt_interaction_json = os.path.join(args.output_dir, "gpt_interaction.json")
    llama_single_json = os.path.join(args.output_dir, "llama_single.json")
    llama_multi_json = os.path.join(args.output_dir, "llama_multi.json")
    llama_interaction_json = os.path.join(args.output_dir, "llama_interaction.json")
    final_json = os.path.join(args.output_dir, "final.json")
    os.makedirs(args.output_dir, exist_ok=True)

    ####################################################################################
    # LOAD LLAMA
    if (args.stage == "") or args.stage == "llama":
        model_id = "neuralmagic/Meta-Llama-3.1-70B-Instruct-quantized.w8a8"
        llm = LLM(
            model=model_id,
            tensor_parallel_size=args.num_gpus,
            max_model_len=8192,
            enforce_eager=True,
            )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=1024)
    ####################################################################################

    # VIDEOS IN BOTH SINGLE/INTERACTION PROCESSED DIRECTORIES
    videos_processed = []
    single_processed_dir = os.path.join(args.processed_dir, "single")
    interaction_processed_dir = os.path.join(args.processed_dir, "interaction")
    for video_id in os.listdir(single_processed_dir):
        if (video_id in os.listdir(interaction_processed_dir)) and (len(os.listdir(interaction_processed_dir)) != 0):
            videos_processed.append(video_id)
    videos_processed = list(set(videos_processed))

    # LOAD DICT IF ALRADY EXISTS : CONTINUE PROCESSING
    gpt_single_dict = load_json(gpt_single_json)
    gpt_interaction_dict = load_json(gpt_interaction_json)
    llama_single_dict = load_json(llama_single_json)
    llama_multi_dict = load_json(llama_multi_json)
    llama_interaction_dict = load_json(llama_interaction_json)
    
    # CHECK IF ALREADY PROCESSED
    already_processed = []
    if os.path.exists(final_json):
        with open(final_json, "r") as f:
            final_dict = json.load(f)
        for video_id in final_dict.keys():
            if all(k in final_dict[video_id] for k in ["gpt_single_object", "gpt_interaction", "llama_single_object", "llama_multi_object", "llama_interaction"]):
                if all(k != None for k in final_dict[video_id].values()):
                    already_processed.append(video_id)
    else:
        final_dict = {}
    
    video_list_to_process = []
    for video_id in videos_processed:
        if video_id not in already_processed:
            video_list_to_process.append(video_id)

    # PROCESS ALL PIPELINE
    count = 0
    passed_not_valid_single = []
    with tqdm(total=len(video_list_to_process)) as pbar:
        for video_id in video_list_to_process:
            
            pbar.set_description(f"PROCESSING VIDEO {video_id}")
            
            if video_id in already_processed:
                pbar.update(1)
                continue
            
            final_dict[video_id] = {}

            count += 1
            if (args.num_samples != 0):
                if count > args.num_samples: break
            
            # SUB-PROCESS CHECK
            process_step = []
            if (args.stage == "") or args.stage == "gpt":
                if video_id not in gpt_single_dict:
                    process_step.append("gpt_single")
                if video_id not in gpt_interaction_dict:
                    process_step.append("gpt_interaction")
            if (args.stage == "") or args.stage == "llama":
                if video_id not in llama_single_dict:
                    process_step.append("llama_single")
                if video_id not in llama_multi_dict:
                    process_step.append("llama_multi")
                if video_id not in llama_interaction_dict:
                    process_step.append("llama_interaction")
            
            # GPT : SINGLE OBJECT CAPTION
            if "gpt_single" in process_step:
                print("Processing GPT Single Object Caption")
                gpt_single_generated_dict = gpt_single_object_caption(os.path.join(single_processed_dir, video_id), args.class_anno_dir)
                gpt_single_dict[video_id] = gpt_single_generated_dict
                final_dict[video_id]["gpt_single_object"] = gpt_single_generated_dict
            else:
                print("Load GPT Single Object Caption")
                final_dict[video_id]["gpt_single_object"] = gpt_single_dict[video_id]
            # SAVE RESULT
            with open(gpt_single_json, "w") as f:
                json.dump(gpt_single_dict, f, indent=4)
            
            ### CHECK AND DELETE OBJECT IF THERE IS NONE VALID SINGLE OBJECT CATIONS
            none_valid = False
            for obj_id, caption in list(final_dict[video_id]["gpt_single_object"].items()):
                none_valid = check_single_caption_valid(caption)
                if none_valid:
                    # TRY AGAIN
                    print("TRY AGAIN...")
                    caption_again = gpt_single_object_caption_again(os.path.join(single_processed_dir, video_id), args.class_anno_dir, obj_id=obj_id)
                    none_valid_again = check_single_caption_valid(caption_again)
                    # CHECK AGAIN
                    if none_valid_again:
                        # REMOVE UNVALID OBJECT CAPTION
                        if obj_id in gpt_single_dict.get(video_id, {}):
                            del gpt_single_dict[video_id][obj_id]
                        if obj_id in final_dict[video_id]["gpt_single_object"]:
                            del final_dict[video_id]["gpt_single_object"][obj_id]
                    else:
                        # SAVE AGAIN
                        gpt_single_dict[video_id][obj_id] = caption_again
                        final_dict[video_id]["gpt_single_object"][obj_id] = caption_again
            # SAVE RESULT AGAIN
            with open(gpt_single_json, "w") as f:
                json.dump(gpt_single_dict, f, indent=4)

            # GPT : INTERACTION CAPTION
            if "gpt_interaction" in process_step:
                print("Processing GPT Interaction Caption")
                gpt_interaction_generated_dict = gpt_interaction_object_caption(os.path.join(interaction_processed_dir, video_id), args.mask_anno_dir)
                gpt_interaction_dict[video_id] = gpt_interaction_generated_dict
                final_dict[video_id]["gpt_interaction"] = gpt_interaction_generated_dict
            else:
                print("Load GPT Interaction Caption")
                final_dict[video_id]["gpt_interaction"] = gpt_interaction_dict[video_id]
            # SAVE RESULT
            with open(gpt_interaction_json, "w") as f:
                json.dump(gpt_interaction_dict, f, indent=4)

            if (args.stage == "") or args.stage == "llama":
                # LLAMA : SINGLE OBJECT CAPTION
                if "llama_single" in process_step:
                    print("Processing LLAMA Single Object Caption")
                    llama_single_generated_dict = llama_single_object_caption(final_dict[video_id]["gpt_single_object"])
                    llama_single_dict[video_id] = llama_single_generated_dict
                    final_dict[video_id]["llama_single_object"] = llama_single_generated_dict
                else:
                    print("Load LLAMA Single Object Caption")
                    final_dict[video_id]["llama_single_object"] = llama_single_dict[video_id]
                # SAVE RESULT
                with open(llama_single_json, "w") as f:
                    json.dump(llama_single_dict, f, indent=4)

                # LLAMA : MULTI OBJECT CAPTION
                if "llama_multi" in process_step:
                    print("Processing LLAMA Multi Object Caption")
                    llama_multi_generated_dict = llama_multi_object_caption(final_dict[video_id]["gpt_single_object"])
                    llama_multi_dict[video_id] = llama_multi_generated_dict
                    final_dict[video_id]["llama_multi_object"] = llama_multi_generated_dict
                else:
                    print("Load LLAMA Multi Object Caption")
                    final_dict[video_id]["llama_multi_object"] = llama_multi_dict[video_id]
                # SAVE RESULT
                with open(llama_multi_json, "w") as f:
                    json.dump(llama_multi_dict, f, indent=4)

                # LLAMA : INTERACTION CAPTION
                if "llama_interaction" in process_step:
                    print("Processing LLAMA Interaction Caption")
                    llama_interaction_generated_dict = llama_interaction_caption(final_dict[video_id]["gpt_single_object"], final_dict[video_id]["gpt_interaction"])
                    llama_interaction_dict[video_id] = llama_interaction_generated_dict
                    final_dict[video_id]["llama_interaction"] = llama_interaction_generated_dict
                else:
                    print("Load LLAMA Interaction Caption")
                    final_dict[video_id]["llama_interaction"] = llama_interaction_dict[video_id]
                # SAVE RESULT
                with open(llama_interaction_json, "w") as f:
                    json.dump(llama_interaction_dict, f, indent=4)

            # SAVE ALL RESULTS
            with open(final_json, "w") as f:
                json.dump(final_dict, f, indent=4)

            if passed_not_valid_single != []:
                print("WARNING!!! There are passed videos")
                print(passed_not_valid_single)

            pbar.update(1)
