import torch
from matplotlib import pyplot as plt
from matplotlib import patches
from PIL import Image, ImageDraw
from torchvision.ops.boxes import box_convert
    
def generate_image_mask(image, bbox, output_path='mask.png'):
    """
    Generate and save an image mask based on bounding boxes and labels.

    Args:
        image (PIL.Image): Original image.
        data (dict): Contains `bboxes` (list of [x1, y1, x2, y2]) and `labels` (list of labels).
        output_path (str): Path to save the mask image (default is 'mask.png').

    Returns:
        None
    """
    # Convert PIL image to NumPy array to get dimensions
    width, height = image.size

    # Create a blank mask
    mask = Image.new("L", (width, height), 0)  # "L" mode for grayscale (0-255)
    draw = ImageDraw.Draw(mask)

    x1, y1, x2, y2 = map(int, bbox)  # Ensure coordinates are integers
    draw.rectangle([x1, y1, x2, y2], fill=255)  # White (255) for the region inside the bbox

    # Save the mask as a PNG file
    return mask
    # mask.save(output_path)
    # print(f"Mask saved to {output_path}")
