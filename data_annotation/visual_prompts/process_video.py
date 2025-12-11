import os
import cv2
import json
import numpy as np
import random
import pycocotools.mask as mask_utils


COLOR_NAME_TO_RGB = {
    'black': (0, 0, 0),
    'white': (255, 255, 255),
    'gray': (128, 128, 128),
    'dark gray': (64, 64, 64),
    'light gray': (200, 200, 200),
    'red': (255, 0, 0),
    'dark red': (139, 0, 0),
    'light red': (255, 102, 102),
    'blue': (0, 0, 255),
    'dark blue': (0, 0, 139),
    'light blue': (173, 216, 230),
    'green': (0, 128, 0),
    'dark green': (0, 100, 0),
    'light green': (144, 238, 144),
    'yellow': (255, 255, 0),
    'light yellow': (255, 255, 153),
    'orange': (255, 165, 0),
    'dark orange': (255, 140, 0),
    'pink': (255, 192, 203),
    'hot pink': (255, 105, 180),
    'purple': (128, 0, 128),
    'light purple': (216, 191, 216),
    'brown': (139, 69, 19),
    'light brown': (181, 101, 29),
    'beige': (245, 245, 220),
    'cream': (255, 253, 208),
    'navy': (0, 0, 128),
    'teal': (0, 128, 128),
    'cyan': (0, 255, 255),
    'light cyan': (224, 255, 255),
    'magenta': (255, 0, 255),
    'lime': (0, 255, 0),
    'olive': (128, 128, 0),
    'coral': (255, 127, 80),
    'salmon': (250, 128, 114),
    'gold': (255, 215, 0),
    'silver': (192, 192, 192),
    'turquoise': (64, 224, 208),
    'lavender': (230, 230, 250),
    'sky blue': (135, 206, 235),
    'forest green': (34, 139, 34),
    'mint': (189, 252, 201),
    'violet': (238, 130, 238),
    'plum': (221, 160, 221),
    'peach': (255, 229, 180),
    'khaki': (240, 230, 140),
    'mustard': (255, 219, 88),
    'charcoal': (54, 69, 79),
    'rose': (255, 228, 225),
}


def generate_fixed_palette(N=10):
    return random.sample(list(COLOR_NAME_TO_RGB.items()), N)


def save_processed_image(frame, output_path="temp.jpg"):
    if frame is None:
        print("[Error] Frame is None, cannot save image.")
        return False
    success = cv2.imwrite(output_path, frame)
    if not success:
        print(f"[Error] Failed to save image to {output_path}")
    return success


def draw_label_with_background(img, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.4,
                               text_color=(255, 255, 255), bg_color=(0, 0, 0), thickness=1, padding=4):
    """Draw label with background box at given position"""
    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    text_width, text_height = text_size
    x, y = position
    top_left = (x, y - text_height - padding)
    bottom_right = (x + text_width + padding, y + baseline)

    cv2.rectangle(img, top_left, bottom_right, bg_color, cv2.FILLED)

    cv2.putText(img, text, (x + padding // 2, y - padding // 2),
                font, font_scale, text_color, thickness, cv2.LINE_AA)


def bbox_overlay_on_frame(frame, bboxes, palette, label=False):
    blended = frame.copy()

    for idx, bbox in enumerate(bboxes):
        if idx >= len(palette):
            print(f"[Warning] Too many boxes in frame, skipping bbox {idx}")
            continue

        color = palette[idx]
        x1, y1, x2, y2 = map(int, bbox)

        # DRAW BBOX
        cv2.rectangle(blended, (x1, y1), (x2, y2), color, thickness=3)

        # LABEL
        if label:
            label_text = f"[{idx}]"
            draw_label_with_background(blended, label_text, (x1, y1 - 5))

    return blended


def mask_overlay_on_frame(frame, masklets, palette, valid_obj_ids=None, label=False):
    blended = frame.copy()
    mask_combined = np.zeros(frame.shape[:2], dtype=np.uint8)
    
    if valid_obj_ids is None:
        for masklet_idx, mask in enumerate(masklets):
            color = palette[masklet_idx]
            if isinstance(mask, dict):
                mask = mask_utils.decode(mask).astype(np.uint8)
            elif isinstance(mask, np.ndarray):
                mask = mask
        
            # MASK OVERLAY
            for c in range(3):
                blended[:, :, c] = np.where(
                    mask > 0,
                    (0.7 * frame[:, :, c] + 0.3 * color[c]).astype(np.uint8),
                    blended[:, :, c]
                )
        
            # OUTLINE
            outline, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(blended, outline, -1, color, thickness=1)
            
            # LABEL
            if label:
                y_indices, x_indices = np.where(mask > 0)
                if len(x_indices) > 0:
                    center_x = int(np.mean(x_indices))
                    center_y = int(np.mean(y_indices))
                    label_text = "[0]"
                    draw_label_with_background(blended, label_text, (center_x, center_y))

    else:
        for masklet_idx, obj_idx in enumerate(valid_obj_ids):
            mask = masklets[masklet_idx]
            color = palette[masklet_idx]
            if isinstance(mask, dict):
                mask = mask_utils.decode(mask).astype(np.uint8)
            elif isinstance(mask, np.ndarray):
                mask = mask
            
            # MASK OVERLAY
            for c in range(3):
                blended[:, :, c] = np.where(
                    mask > 0,
                    (0.7 * frame[:, :, c] + 0.3 * color[c]).astype(np.uint8),
                    blended[:, :, c]
                )
            
            # OUTLINE
            outline, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(blended, outline, -1, color, thickness=1)
            
            # LABEL
            if label:
                y_indices, x_indices = np.where(mask > 0)
                if len(x_indices) > 0:
                    center_x = int(np.mean(x_indices))
                    center_y = int(np.mean(y_indices))
                    label_text = f"[{obj_idx}]"
                    draw_label_with_background(blended, label_text, (center_x, center_y))

    return blended


def mask_outline_on_frame(frame, masklets, palette, label=False):
    blended = frame.copy()
    mask_combined = np.zeros(frame.shape[:2], dtype=np.uint8)

    for masklet_idx, mask in enumerate(masklets):
        color = palette[masklet_idx]
        if isinstance(mask, dict):
            mask = mask_utils.decode(mask).astype(np.uint8)
        elif isinstance(mask, np.ndarray):
            mask = mask

        # OUTLINE
        outline, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, outline, -1, color, thickness=2)

        # LABEL
        if label:
            y_indices, x_indices = np.where(mask > 0)
            if len(x_indices) > 0:
                center_x = int(np.mean(x_indices))
                center_y = int(np.mean(y_indices))
                label_text = f"[{masklet_idx}]"
                draw_label_with_background(blended, label_text, (center_x, center_y))

    return blended


def blur_frame(frame, masklets=None, bboxes=None):

    # blur_strength, sigmaX = (25, 25), 0
    blur_strength, sigmaX = (51, 51), 25
    # blur_strength, sigmaX = (75, 75), 50

    full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    
    if masklets is not None:
        for masklet_idx, mask in enumerate(masklets):
            if isinstance(mask, dict):
                mask = mask_utils.decode(mask).astype(np.uint8)
            elif isinstance(mask, np.ndarray):
                mask = mask

            full_mask = np.maximum(full_mask, mask)

        blurred_frame = cv2.GaussianBlur(frame, blur_strength, sigmaX=sigmaX)

        result = np.where(full_mask[:, :, np.newaxis] == 255, frame, blurred_frame)
    
    elif bboxes is not None:
        blurred_frame = cv2.GaussianBlur(frame, blur_strength, sigmaX=sigmaX)
        result = blurred_frame.copy()

        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            full_mask[y1:y2, x1:x2] = 1

        result = np.where(full_mask[:, :, np.newaxis] == 1, frame, blurred_frame)

    else:
        raise ValueError("No masklets or bboxes are given!")

    return result


def greyscale_frame(frame, masklets=None, bboxes=None):
    full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    if masklets is not None:
        for masklet in masklets:
            if isinstance(masklet, dict):
                mask = mask_utils.decode(masklet).astype(np.uint8)
            elif isinstance(masklet, np.ndarray):
                mask = masklet.astype(np.uint8)
            else:
                raise TypeError("Unsupported masklet type.")
            full_mask = np.maximum(full_mask, mask)

    elif bboxes is not None:
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            full_mask[y1:y2, x1:x2] = 1
    
    else:
        raise ValueError("No masklets or bboxes are given!")

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame_3ch = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

    if full_mask.max() == 1:
        full_mask = full_mask * 255

    result = np.where(full_mask[:, :, np.newaxis] == 255, frame, gray_frame_3ch)

    return result


def black_background_frame(frame, masklets=None, bboxes=None):
    full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    if masklets is not None:
        for masklet in masklets:
            if isinstance(masklet, dict):
                mask = mask_utils.decode(masklet).astype(np.uint8)
            elif isinstance(masklet, np.ndarray):
                mask = masklet.astype(np.uint8)
            else:
                raise TypeError("Unsupported masklet type.")
            full_mask = np.maximum(full_mask, mask)

    elif bboxes is not None:
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            full_mask[y1:y2, x1:x2] = 1

    else:
        raise ValueError("No masklets or bboxes are given!")

    if full_mask.max() == 1:
        full_mask = full_mask * 255

    black_background = np.zeros_like(frame)
    result = np.where(full_mask[:, :, np.newaxis] == 255, frame, black_background)

    return result


def bbox_crop(frame, bbox):
    """
    Crop a region from the frame based on the given bbox [x1, y1, x2, y2].
    If bbox is invalid (e.g., all zeros), returns a black image with same shape as bbox area.
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    if x1 == x2 or y1 == y2 or (x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0):
        # Return black image with default size if bbox is invalid
        h, w = frame.shape[:2]
        dummy_crop = np.zeros((64, 64, 3), dtype=np.uint8)  # or choose average crop size
        return dummy_crop

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)

    cropped = frame[y1:y2, x1:x2]
    return cropped


def resize_to_even(img):
    h, w = img.shape[:2]
    new_h = h if h % 2 == 0 else h + 1
    new_w = w if w % 2 == 0 else w + 1
    if new_h != h or new_w != w:
        img = cv2.resize(img, (new_w, new_h))
    return img


def get_bbox_from_mask(binary_mask):
    ys, xs = np.where(binary_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None  # empty mask
    x_min = int(xs.min())
    y_min = int(ys.min())
    x_max = int(xs.max())
    y_max = int(ys.max())
    return [x_min, y_min, x_max, y_max]
