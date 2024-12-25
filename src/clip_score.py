import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import List, Tuple
import os
from collections import defaultdict
import tqdm

def calculate_clip_text_scores_folder(image_path_list: List[str], text: str) -> List[Tuple[str, float]]:
    # Load the CLIP model and processor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    # Check if the number of images is equal to 100
    # if len(image_paths) != 10:
    #     raise ValueError(f"The number of images in the folder must be exactly 25. Found {len(image_paths)} images.")

    # Process images in batches to avoid memory issues
    batch_size = 32
    all_scores = []

    pbar = tqdm.tqdm(range(0, len(image_path_list), batch_size))

    for i in pbar:
        batch_paths = image_path_list[i:i+batch_size]
        
        # Load and preprocess the images
        images = [Image.open(path).convert('RGB') for path in batch_paths]
        inputs = processor(text=[text], images=images, return_tensors="pt", padding=True).to(device)

        # Calculate the CLIP scores
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            clip_scores = logits_per_image.squeeze().tolist()

        # Handle single image case
        if isinstance(clip_scores, float):
            clip_scores = [clip_scores]

        # Pair each image path with its score
        all_scores.extend(clip_scores)

        pbar.set_description(f"Processing clip text scores")

    return all_scores



def load_images_from_folder(folder_path: str) -> List[Tuple[str, Image.Image]]:
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    images = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(image_extensions):
            file_path = os.path.join(folder_path, filename)
            try:
                with Image.open(file_path) as img:
                    images.append((file_path, img.convert('RGB')))
            except IOError:
                print(f"Error opening {file_path}. Skipping.")


    return images

def calculate_clip_image_scores(
        input_images: List[Image.Image], 
        reference_images: List[Tuple[str, Image.Image]],
    ) -> List[Tuple[str, float]]:
    # Load the CLIP model and processor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    score_list = []

    pbar = tqdm.tqdm(range(0, len(input_images)))

    for i in pbar:
        input_image = input_images[i]
        
        # Preprocess input images
        inputs = processor(images=input_image, return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            # Get input image features
            input_features = model.get_image_features(**inputs).detach().cpu()
            
            single_score_list = []

            # Calculate scores for each reference image
            for _, ref_img in reference_images:
                ref_inputs = processor(images=[ref_img], return_tensors="pt", padding=True).to(device)
                ref_features = model.get_image_features(**ref_inputs).detach().cpu()
                
                # Calculate similarity scores
                similarity = 100 * torch.nn.functional.cosine_similarity(input_features, ref_features)
                scores = similarity.squeeze().tolist()
                
                single_score_list.append(scores)
        
        score_list.append(sum(single_score_list) / len(single_score_list))
                
        pbar.set_description(f"Processing clip image scores")

    return score_list


def calculate_clip_image_scores_folder(image_path_list: List[str], reference_folder: str) -> float:

    input_images = [Image.open(image_path) for image_path in image_path_list]    

    reference_images = []
    for ref_folder in reference_folder:
        reference_images.extend(load_images_from_folder(ref_folder))
        
    avg_scores = calculate_clip_image_scores(input_images, reference_images)

    return avg_scores