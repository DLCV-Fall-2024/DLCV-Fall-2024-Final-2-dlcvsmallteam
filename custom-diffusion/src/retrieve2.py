from clip_retrieval.clip_back import load_clip_indices, KnnService, ClipOptions, MetadataService,ParquetMetadataProvider
import base64
from PIL import Image
import json
from datetime import datetime


indice_folder = "./improved_aesthetics_6.5plus_clip_retrieval"
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
columns = ["url", "caption"]
resources = load_clip_indices("./indices_paths.json", clip_options)
knn_service = KnnService(clip_resources=resources)

output_dir = "./result"
input_image = "example/cat.jpg"
import os

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
with open(input_image, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

img_results = []

img_results = knn_service.query(
    image_input=encoded_string,  # img2img
    # text_input="cat",  #txt2img
    modality="image",
    deduplicate=True,
    # image_url_input="https://s3.bmp.ovh/imgs/2024/10/10/9889c696e71bcbbf.jpeg", #imgurl2img
    num_result_ids=10,
)
# print(img_results)
file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"retrieval_results_{file_timestamp}.json"
with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
    json.dump(img_results, f, ensure_ascii=False, indent=4)
for js in img_results:
    print(js)
    
id_values = [item["id"] for item in img_results]
metadata = ParquetMetadataProvider(os.path.join(indice_folder,"metadata"))
image_meta_data = metadata.get(id_values)
for item in image_meta_data:
    print(item)
    

source_root = "akameswa/improved_aesthetics_6.5plus_webdataset_unzip"

merged_source_json = []
for item in image_meta_data:
    json_path = os.path.join(source_root, item["image_path"] + ".json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        print(data)
        merged_source_json.extend(data if isinstance(data, list) else [data])

filename_source = f"source_results_{file_timestamp}.json"
with open(os.path.join(output_dir, filename_source), "w", encoding="utf-8") as f:
    json.dump(merged_source_json, f, ensure_ascii=False, indent=4)

images = [
    Image.open(os.path.join(source_root, item["image_path"] + ".jpg"))
    for item in image_meta_data
]

