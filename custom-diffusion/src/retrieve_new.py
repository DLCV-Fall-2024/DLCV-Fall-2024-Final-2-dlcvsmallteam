# Copyright 2022 Adobe Research. All rights reserved.
# To view a copy of the license, visit LICENSE.md.

import argparse
import os
import tqdm
from PIL import Image
import torch
from clip_retrieval.clip_back import load_clip_indices, KnnService, ClipOptions, MetadataService,ParquetMetadataProvider


def retrieve(target_name, outpath, num_class_images):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    indice_folder = "/tmp2/seanfu/improved_aesthetics_6.5plus_clip_retrieval"

    clip_options = ClipOptions(
    indice_folder=indice_folder,
    clip_model="open_clip:ViT-B-32/laion2b_s34b_b79k",
    enable_hdf5=False,
    enable_faiss_memory_mapping=False,
    columns_to_return=["url", "caption"],
    reorder_metadata_by_ivf_index=False,
    enable_mclip_option=False,
    use_jit=False,
    use_arrow=False,
    provide_safety_model=False,
    provide_violence_detector=False,
    provide_aesthetic_embeddings=False,
    )
    
    resources = load_clip_indices("./indices_paths.json", clip_options)
    knn_service = KnnService(clip_resources=resources)

    num_images = 2*num_class_images

    img_results = knn_service.query(
        # image_input=encoded_string,  # img2img
        text_input=target_name,  #txt2img
        modality="image",
        deduplicate=True,
        # image_url_input="https://s3.bmp.ovh/imgs/2024/10/10/9889c696e71bcbbf.jpeg", #imgurl2img
        num_result_ids=num_images,
    )
    
    metadata = ParquetMetadataProvider(os.path.join(indice_folder,"metadata"))
    
    if len(target_name.split()):
        target = '_'.join(target_name.split())
    else:
        target = target_name
    os.makedirs(f'{outpath}/{target}', exist_ok=True)

    count = 0
    captions = []
    
    img_results_caption = [img_result for img_result in img_results if 'caption' in img_result]
    
    for each in tqdm.tqdm(img_results_caption):
        
        name = f'{outpath}/{target}/{count}.jpg'
        
        id_value = metadata.get([each["id"]])
        
        source_root = f"/tmp2/seanfu/improved_aesthetics_6.5plus_webdataset"
        
        img = Image.open(f"{source_root}/{id_value[0]['image_path']}.jpg")
        
        img.save(name, 'JPEG')
        
        captions.append(each['caption'])
        
        count += 1
        
    print(outpath)

    with open(f'{outpath}/caption.txt', 'w') as f:
        for each in captions:
            f.write(each.strip() + '\n')

    with open(f'{outpath}/images.txt', 'w') as f:
        for p in range(count):
            f.write(f'{outpath}/{target}/{p}.jpg' + '\n')


def parse_args():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--target_name', help='target string for query',
                        type=str)
    parser.add_argument('--outpath', help='path to save retrieved images', default='./',
                        type=str)
    parser.add_argument('--num_class_images', help='number of retrieved images', default=10,
                        type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    retrieve(args.target_name, args.outpath, args.num_class_images)
