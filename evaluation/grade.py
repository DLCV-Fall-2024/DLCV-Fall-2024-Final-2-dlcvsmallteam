import json
import argparse
import os
from clip_image import calculate_clip_image_scores_folder
from clip_text import calculate_clip_text_scores_folder

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process some JSON data.')
    parser.add_argument('--json_path', type=str, help='Path to the JSON file')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory of saved output')
    parser.add_argument('-s', '--setting', type=int, default=0, help='Setting of the evaluation')
    parser.add_argument('-d', '--detail', action='store_true', help='Print detailed information')
    # Parse the arguments
    args = parser.parse_args()
    json_path = args.json_path
    output_dir = args.output_dir
    
    assert args.setting in [-1, 0, 1, 2, 3], "Setting must be 0, 1, 2, or 3"
    
    # Assuming the JSON is saved in a file named 'data.json'
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    all_input_dirs = [
        ['Data/concept_image/cat2', 'Data/concept_image/dog6'],
        ['Data/concept_image/flower_1', 'Data/concept_image/vase'],
        ['Data/concept_image/dog', 'Data/concept_image/pet_cat1', 'Data/concept_image/dog6'],
        ['Data/concept_image/cat2', 'Data/concept_image/wearable_glasses', 'Data/concept_image/watercolor']
    ]

    pass_count = 0
    count = 0
    print("\n===============================================start evaluation================================================\n")

    # Iterate through the data and print the prompt_4_clip_eval
    
    if args.setting == -1:
        
        clip_t_list = []
        clip_i_list = []
        
        for setting in [0,1,2,3]:
            key, value = list(data.items())[setting]
            
            input_folder_path = all_input_dirs[setting]
            output_folder_path = os.path.join(output_dir, str(setting))

            src = value['src_image']
            
            clip_eval = value['prompt_4_clip_eval']

            print(f"Image source: \"{src}\", text prompt: {clip_eval}")

            image_scores = calculate_clip_image_scores_folder(output_folder_path, input_folder_path)
            text_scores = calculate_clip_text_scores_folder(output_folder_path, clip_eval)
            
            # total_score = image_scores + 2.5 * text_scores
            image_score = sum(image_scores) / len(image_scores)
            text_score = sum(text_scores) / len(text_scores)

            total_scores = [i + 2 * t for i, t in zip(image_scores, text_scores)]

            sorted_indices = sorted(range(len(total_scores)), key=lambda k: total_scores[k])[-1:]

            image_scores_filtered = [image_scores[i] for i in sorted_indices]
            text_scores_filtered = [text_scores[i] for i in sorted_indices]

            print(f"CLIP Image Score: {sum(image_scores_filtered)/len(image_scores_filtered):.2f}")
            print(f"CLIP Text Score: {sum(text_scores_filtered)/len(text_scores_filtered):.2f}")

            if args.detail:
                print(f"CLIP Image Scores: {image_scores_filtered}")
                print(f"CLIP Text Scores: {text_scores_filtered}")
            
            clip_t_list.append(sum(text_scores_filtered)/len(text_scores_filtered))
            clip_i_list.append(sum(image_scores_filtered)/len(image_scores_filtered))
            
        print(f"\nAverage CLIP Image Score: {sum(clip_i_list) / len(clip_i_list):.2f}")
        print(f"Average CLIP Text Score: {sum(clip_t_list) / len(clip_t_list):.2f}")
        
    else:
    
        key, value = list(data.items())[args.setting]
        
        input_folder_path = all_input_dirs[args.setting]
        output_folder_path = os.path.join(output_dir, str(args.setting))

        src = value['src_image']
        
        clip_eval = value['prompt_4_clip_eval']

        print(f"Image source: \"{src}\", text prompt: {clip_eval}")

        image_scores = calculate_clip_image_scores_folder(output_folder_path, input_folder_path)
        text_scores = calculate_clip_text_scores_folder(output_folder_path, clip_eval)
        
        # total_score = image_scores + 2.5 * text_scores
        image_score = sum(image_scores) / len(image_scores)
        text_score = sum(text_scores) / len(text_scores)

        print(f"CLIP Image Score: {image_score:.2f}")
        print(f"CLIP Text Score: {text_score:.2f}")

        if args.detail:
            print(f"CLIP Image Scores: {image_scores}")
            print(f"CLIP Text Scores: {text_scores}")

    