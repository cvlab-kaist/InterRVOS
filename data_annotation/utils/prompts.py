import os
import json
from typing import List, Dict, Any, Literal


def make_prompt(
    stage: Literal[
        "single_object_caption",
        "object_interaction",
        ],
    **kwargs
    ) -> str:


    # SINGLE OBJECT CAPTION
    if stage == "single_object_caption":
        prompt = f"""
<task>
You are given a video where specific objects are highlighted. Your task is to describe only the highlighted object, focusing on both its visual appearance and how it moves or changes position throughout the video.
</task>

<objectives>
1. Provide a **localized caption** that describes:
   - The **visual appearance** (color, shape, texture, category, etc.) of the highlighted object.
   - The object's **motion** or **spatial movement** (e.g., moving left, jumping, rotating).
2. Do not mention any other objects that are not highlighted.
3. Use only the information that can be **visually confirmed** from the video. **Do not infer or assume anything** that is not clearly visible (e.g., names of people, unobservable intent or unseen background).
4. **Do not refer to the red highlight, colored contour, or any visual marking used to identify the object.** Focus only on the object’s inherent visual and behavioral properties.
5. Use clear, concise language that reflects what is visually and spatially observable from the highlighted object only.
6. The object's motion description must refer to **the same highlighted object** whose appearance you just described. Do not describe movement of unrelated objects, background elements, or the overall scene.
7. If the highlighted object is stationary or only slightly moving, describe that accurately. Do not fabricate or exaggerate movement based on nearby motion.
</objectives>

<inputDetails>
- The input is a short video clip containing multiple objects.
- One or more objects are highlighted using a **red-colored contour around their boundary**.
- The video is designed to preserve the object's **appearance** and provide visual cues for its **motion** across frames.
- Focus only on the object with the **red-colored boundary**, but do **not** describe the boundary or outline itself in your output.
</inputDetails>

<objectClass>
- The object class is "{kwargs["obj_class"]}".
- Use this information only to support your understanding of what kind of object to describe.
- However, you must describe **the object that is visually highlighted** in the video (e.g., marked with a red boundary or mask).
- If there are multiple objects of the same class in the scene, **focus solely on the highlighted one**, even if others appear more salient or central.
</objectClass>

<outputFormat>
Provide **two distinct sentences** in a single paragraph form:
1. Describe what the object looks like (e.g., "A small brown dog with curly fur and a blue collar.")
2. Describe how the object moves or behaves in the video (e.g., "It runs from left to right across the grassy field, occasionally looking back.")
Avoid describing things that cannot be visually confirmed from the video.
</outputFormat>
"""

    # CHECK OBJECT INTERACTION
    elif stage == "object_interaction":
        prompt = f"""
<task>
You are given a video in which multiple labeled objects appear. Your task is to identify any visible interaction between the labeled objects, determine the type and direction of interaction, and describe it appropriately.
</task>

<objectives>
1. Determine whether any interaction is visually observable between the labeled objects.
2. If yes, classify the interaction as:
   - "bidirectional" (e.g., mutual interaction like "[2] and [3] are dancing together")
   - "unidirectional" (e.g., directional interaction like "[0] is handing something to [1]")
3. For each interaction:
   - If bidirectional → provide **one sentence** describing the mutual interaction.
   - If unidirectional → provide **two sentences**:
     - One where the **initiator** is the subject
     - One where the **receiver** is the subject (in passive form)
   - **Include all objects that are directly or indirectly involved in the interaction in the `object_pair` list.**
   - **If the interaction is `unidirectional`, provide one sentence for each object in `object_pair`, using that object as the grammatical subject.**
     - For example, if `object_pair = ["[0]", "[1]", "[7]"]`, there should be three sentences:
       - One with [0] as the subject
       - One with [1] as the subject
       - One with [7] as the subject
4. Interactions involving more than two objects (e.g., [0], [1], [2]) should be described as a group if they jointly participate in the same action.
5. Always refer to objects using their exact labels like "[1]", "[2]", etc.
6. Only describe interactions that are visually verifiable—do not infer hidden intentions, emotions, or relationships.
</objectives>

<inputDetails>
- The input video contains labeled objects with the following identifiers:
  **{kwargs["valid_obj_ids"]}**
- These are the only valid object labels. You must not use or invent any other object identifiers.
- Each object is highlighted with a colored outline.
</inputDetails>

<additionalInput>
The following object categories are provided as prior knowledge:

obj_categories = {kwargs["obj_categories"]}

These categories may guide your understanding of plausible interactions, but your final decisions must rely strictly on visual evidence.
</additionalInput>

<reasoningSteps>
Step-by-step reasoning:
1. Consider only the labeled objects: {kwargs["valid_obj_ids"]}
2. Do not assume the existence of any other object labels (e.g., [0], [3] are invalid).
3. Examine all valid pairs and groups of the provided objects.
4. For each candidate interaction:
   a. Observe their motion, spatial alignment, and relative timing.
   b. If interaction occurs:
      i. Classify it as bidirectional or unidirectional.
      ii. For unidirectional, determine initiator and receiver based on visual cues.
   c. After writing the descriptions:
    - Ensure that every object in `object_pair` appears as the **grammatical subject** of at least one sentence.
5. Construct appropriate descriptions accordingly.
6. If no interactions are observed, return interaction = "NO".
</reasoningSteps>

<outputFormat>
{{
  "interaction": "YES" or "NO",
  "interactions": [
    {{
      "object_pair": ["[1]", "[2]"],
      "type": "bidirectional",
      "descriptions": [
        "Object [1] and object [2] are shaking hands."
      ]
    }},
    {{
      "object_pair": ["[8]", "[2]"],
      "type": "unidirectional",
      "descriptions": [
        "Object [8] is pointing at object [2].",
        "Object [2] is being pointed at by object [8]."
      ]
    }}
  ] or None
}}
</outputFormat>

<exampleOutput>
Example 1: Interaction Present
{{
  "interaction": "YES",
  "interactions": [
    {{
      "object_pair": ["[1]", "[2]"],
      "type": "bidirectional",
      "descriptions": [
        "Object [1] and object [2] are holding hands while walking together."
      ]
    }}
  ]
}}

Example 2: Interaction Present (more than two objects)
{{
  "interaction": "YES",
  "interactions": [
    {{
      "object_pair": ["[0]", "[7]", "[1]"],
      "type": "unidirectional",
      "descriptions": [
        "Object [1] is touching object [0] with object [7].",
        "Object [0] is being touched by object [1] with object [7].",
        "Object [7] is being used by object [1] to feed object [0]."
      ]
    }}
  ]
}}

Example 3: No Interaction
{{
  "interaction": "NO",
  "interactions": None
}}
</exampleOutput>

<selfCheck>
Before finalizing your output:
Before finalizing your output:
- Double-check that every object mentioned in the descriptions is present in the object_pair.
- Double-check that each object in the object_pair appears as the grammatical **subject** in at least one sentence.
</selfCheck>

<avoid>
- Do not describe object appearance or identity (e.g., "the man", "the woman")
- Do not reference unlabeled object identifiers (e.g., [0], [3]) that are not listed above.
- Do not guess context, background, or emotional intent.
- If unsure about any interaction, default to "NO".
</avoid>
"""

    else:
        raise ValueError(f"Invalid stage: {stage}")

    return prompt
