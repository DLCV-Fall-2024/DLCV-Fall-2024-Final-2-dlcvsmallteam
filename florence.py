from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image, ImageDraw, ImageFont
import random
import requests
import copy
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches

model_id = 'microsoft/Florence-2-large-ft'
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).eval().cuda()
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

def run_example(task_prompt, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    print(f"inputs ---> {inputs}")

    generated_ids = model.generate(
      input_ids=inputs["input_ids"].cuda(),
      pixel_values=inputs["pixel_values"].cuda(),
      max_new_tokens=1024,
      early_stopping=False,
      do_sample=False,
      num_beams=3,
    )
    print(f"generated_ids ---> {generated_ids}")

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    print(f"generated_text ---> {generated_text}")

    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )

    return parsed_answer

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
image = Image.open(requests.get(url, stream=True).raw)

image = Image.open('./output/inpainting_images/0/refined_image.png')

# task_prompt = '<REFERRING_EXPRESSION_SEGMENTATION>'
task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
results = run_example(task_prompt, text_input="dog")
print(results)

colormap = ['blue','orange','green','purple','brown','pink','gray','olive','cyan','red',
            'lime','indigo','violet','aqua','magenta','coral','gold','tan','skyblue']

def plot_bbox(image, data):

    # Display the image
    # plt.figure(figsize=(image.shape[1] / 100, image.shape[0] / 100), dpi=100)
    plt.imshow(image)
    

    # Plot each bounding box
    for bbox, label in zip(data['bboxes'], data['labels']):
        # Unpack the bounding box coordinates
        x1, y1, x2, y2 = bbox
        # Create a Rectangle patch
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
        # Add the rectangle to the Axes
        plt.gca().add_patch(rect)
        # Annotate the label
        plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))

    # Remove the axis ticks and labels
    plt.axis('off')

    # Show the plot
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig('output.png', bbox_inches='tight', pad_inches=0)

def generate_and_save_mask(image, data, output_path='mask.png'):
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

    # Draw rectangles for each bounding box
    for bbox in data['bboxes']:
        x1, y1, x2, y2 = map(int, bbox)  # Ensure coordinates are integers
        draw.rectangle([x1, y1, x2, y2], fill=255)  # White (255) for the region inside the bbox

    # Save the mask as a PNG file
    mask.save(output_path)
    print(f"Mask saved to {output_path}")

def draw_polygons(image, prediction, fill_mask=False):
    """
    Draws segmentation masks with polygons on an image.

    Parameters:
    - image_path: Path to the image file.
    - prediction: Dictionary containing 'polygons' and 'labels' keys.
                  'polygons' is a list of lists, each containing vertices of a polygon.
                  'labels' is a list of labels corresponding to each polygon.
    - fill_mask: Boolean indicating whether to fill the polygons with color.
    """
    # Load the image

    draw = ImageDraw.Draw(image)


    # Set up scale factor if needed (use 1 if not scaling)
    scale = 1

    # Iterate over polygons and labels
    for polygons, label in zip(prediction['polygons'], prediction['labels']):
        color = random.choice(colormap)
        fill_color = random.choice(colormap) if fill_mask else None

        for _polygon in polygons:
            _polygon = np.array(_polygon).reshape(-1, 2)
            if len(_polygon) < 3:
                print('Invalid polygon:', _polygon)
                continue

            _polygon = (_polygon * scale).reshape(-1).tolist()

            # Draw the polygon
            if fill_mask:
                draw.polygon(_polygon, outline=color, fill=fill_color)
            else:
                draw.polygon(_polygon, outline=color)

            # Draw the label text
            draw.text((_polygon[0] + 8, _polygon[1] + 2), label, fill=color)

    # Save or display the image
    #image.show()  # Display the image
    image.save('output.png')  # Save the image

output_image = copy.deepcopy(image)

if task_prompt == '<REFERRING_EXPRESSION_SEGMENTATION>':
    draw_polygons(output_image, results['<REFERRING_EXPRESSION_SEGMENTATION>'], fill_mask=True)
else:
    plot_bbox(image, results['<CAPTION_TO_PHRASE_GROUNDING>'])
    generate_and_save_mask(image, results['<CAPTION_TO_PHRASE_GROUNDING>'], output_path='mask.png')

