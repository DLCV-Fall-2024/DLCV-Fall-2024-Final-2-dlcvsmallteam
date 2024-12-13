from GroundingDINO.groundingdino.util.inference import load_image, load_model, predict, annotate
import cv2
import numpy as np
import torch
from torchvision.ops import box_convert
import xml.etree.ElementTree as ET
from diffusers import StableDiffusionXLInpaintPipeline

from src.utils.xml_utils import get_tags_from_xml_file
from PIL import Image
import os

def generate_masks_with_grounding(image_source, boxes):
    h, w, _ = image_source.shape
    boxes_unnorm = boxes * torch.Tensor([w, h, w, h])
    boxes_xyxy = box_convert(boxes=boxes_unnorm, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    mask = np.zeros_like(image_source)
    for box in boxes_xyxy:
        x0, y0, x1, y1 = box
        mask[int(y0):int(y1), int(x0):int(x1), :] = 255
    return mask

def area(box):
    
    box_xyxy = box_convert(boxes=box, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    area = (box_xyxy[2] - box_xyxy[0]) * (box_xyxy[3] - box_xyxy[1])
    
    return area

def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) between two boxes.
    Each box is represented as [x_min, y_min, x_max, y_max].
    """

    box1_convert = box_convert(boxes=box1, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    box2_convert = box_convert(boxes=box2, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    
    x_min1, y_min1, x_max1, y_max1 = box1_convert
    x_min2, y_min2, x_max2, y_max2 = box2_convert
    
    print(x_min1, y_min1, x_max1, y_max1)
    print(x_min2, y_min2, x_max2, y_max2)

    # Compute the intersection
    inter_x_min = max(x_min1, x_min2)
    inter_y_min = max(y_min1, y_min2)
    inter_x_max = min(x_max1, x_max2)
    inter_y_max = min(y_max1, y_max2)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

    # Compute the union
    box1_area = (x_max1 - x_min1) * (y_max1 - y_min1)
    box2_area = (x_max2 - x_min2) * (y_max2 - y_min2)

    # union_area = box1_area + box2_area - inter_area
    union_area = min(box1_area, box2_area)

    if union_area == 0:
        return 0

    return inter_area / union_area


def processing_masks(boxes, logits, phrases):

    unique_phrases = list(set(phrases))
    filtered_boxes = []
    filtered_logits = []
    filtered_phrases = []

    for phrase in unique_phrases:
        # Get indices of the current phrase
        indices = [i for i, p in enumerate(phrases) if p == phrase]

        area_list = [area(boxes[i]) for i in indices]
        
        sorted_data = sorted(zip(indices, area_list))
        
        sorted_indices, sorted_area_list = zip(*sorted_data)
        
        print(sorted_indices, sorted_area_list)

        if not indices:
            continue

        # Sort indices by logits in descending order
        indices = sorted(indices, key=lambda i: logits[i], reverse=True)

        # Keep the largest box (highest logit)
        keep_box = boxes[indices[0]]
        keep_logit = logits[indices[0]]

        filtered_boxes.append(keep_box)
        filtered_logits.append(keep_logit)
        filtered_phrases.append(phrase)

        # Remove boxes with high IoU overlap
        for idx in sorted_indices[1:]:
            iou = compute_iou(keep_box, boxes[idx])
            print('iou:', iou)
            if iou < 0.5:  # Threshold for mIoU
                filtered_boxes.append(boxes[idx])
                filtered_logits.append(logits[idx])
                filtered_phrases.append(phrase)

    return torch.stack(filtered_boxes), filtered_logits, filtered_phrases

model = load_model(
    './GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py',
    './groundingdino_swinb_cogcoor.pth'
)

image_path = 'test.png'
xml_file = './xmls/0.xml'
concepts = get_tags_from_xml_file(xml_file)

print("Concepts in the XML file: ", concepts)

for i, (depth, tags) in enumerate(concepts.items()):
    
    if i == 0:
        continue
    
    print(f"Layer {depth}: {tags}")

    text_prompt = ' . '.join(tags)
    box_threshold = 0.35
    text_threshold = 0.35

    image_source, image = load_image(image_path)

    print(image.size())
    print(image_source.shape)

    boxes, logits, phrases = predict(
        model,
        image,
        text_prompt,
        box_threshold,
        text_threshold
    )
    
    print(boxes, logits, phrases)
    
    boxes, logits, phrases = processing_masks(boxes, logits, phrases)
    
    print(boxes, logits, phrases)

    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    cv2.imwrite('test_annotated.png', annotated_frame)

    for idx, phrase in enumerate(phrases):
        print(phrase)

        image_mask = generate_masks_with_grounding(image_source, boxes[idx].unsqueeze(0))

        # Save the mask as an image
        cv2.imwrite(f'test_mask_{phrase}.png', image_mask)
        
        pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
            # 'diffusers/stable-diffusion-xl-1.0-inpainting-0.1',
            'stabilityai/stable-diffusion-xl-base-1.0',
            torch_dtype=torch.float16
        ).to('cuda')
        
        if idx == 0:
            pipeline.load_lora_weights(os.path.join(os.getcwd(),
                'lora-trained-xl/cat2/checkpoint-500/pytorch_lora_weights.safetensors')
            )
        else:
            pipeline.load_lora_weights(os.path.join(os.getcwd(),
                'lora-trained-xl/dog6/checkpoint-500/pytorch_lora_weights.safetensors')
            )
            
        if idx == 0:
            origin_image = Image.open('test.png').convert('RGB')
        else:
            origin_image = Image.open('test_inpainting.png').convert('RGB')
            
        mask_image = Image.open(f'test_mask_{phrase}.png').convert('L')
        
        if idx == 0:
            prompt = 'A sks cat.'
        else:
            prompt = 'A sks dog.'
            
        result = pipeline(
            prompt=prompt,
            image=origin_image,
            mask_image=mask_image,
        ).images[0]
        
        result.save(f'test_inpainting.png')