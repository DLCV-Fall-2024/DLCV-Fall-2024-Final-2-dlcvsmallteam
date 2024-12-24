#This is an example that uses the websockets api to know when a prompt execution is done
#Once the prompt execution is done it downloads the images using the /history endpoint

import websocket #NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import json
import urllib.request
import urllib.parse
import requests
import os
import sys
server_address = "127.0.0.1:8188"
client_id = str(uuid.uuid4())

def queue_prompt(prompt):
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req =  urllib.request.Request("http://{}/prompt".format(server_address), data=data)
    return json.loads(urllib.request.urlopen(req).read())

def get_image(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen("http://{}/view?{}".format(server_address, url_values)) as response:
        return response.read()

def get_history(prompt_id):
    with urllib.request.urlopen("http://{}/history/{}".format(server_address, prompt_id)) as response:
        return json.loads(response.read())

def get_images(ws, prompt):
    prompt_id = queue_prompt(prompt)['prompt_id']
    output_images = {}
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break #Execution is done
        else:
            continue #previews are binary data

    history = get_history(prompt_id)[prompt_id]
    for o in history['outputs']:
        for node_id in history['outputs']:
            node_output = history['outputs'][node_id]
            if 'images' in node_output:
                images_output = []
                for image in node_output['images']:
                    image_data = get_image(image['filename'], image['subfolder'], image['type'])
                    images_output.append(image_data)
            output_images[node_id] = images_output

    return output_images


def upload_file(file, subfolder="", overwrite=False):
    try:
        # Wrap file in formdata so it includes filename
        body = {"image": file}
        data = {}
        
        if overwrite:
            data["overwrite"] = "true"
  
        if subfolder:
            data["subfolder"] = subfolder

        resp = requests.post(f"http://{server_address}/upload/image", files=body,data=data)
        
        if resp.status_code == 200:
            data = resp.json()
            # Add the file to the dropdown list and update the widget value
            path = data["name"]
            if "subfolder" in data:
                if data["subfolder"] != "":
                    path = data["subfolder"] + "/" + path
            

        else:
            print(f"{resp.status_code} - {resp.reason}")
    except Exception as error:
        print(error)
    return path

lora_paths = {
    "pet_cat": ["pet_cat_rank4_bf16-step00640.safetensors", 1.5, 1.5],
    "dog": ["dog_rank4_bf16-step00640.safetensors", 1.0, 1.0],
    "dog6": ["dog6_rank4_bf16.safetensors", 1.5, 1.5],
    "cat2": ["cat2_rank4_bf16.safetensors", 1.5, 1.5],
    "flower_1": ["flower_1_rank4_bf16.safetensors", 1.0, 1.0],
    "vase": ["vase_rank4_bf16.safetensors", 1.5, 1.5],
    "wearable_glasses": ["wearable_glasses_rank4_bf16.safetensors", 1.0, 1.0]
}

#load workflow from file
with open("Flux_Inpaint_API_Pet_Cat.json", "r", encoding="utf-8") as f:
    workflow_data = f.read()

workflow = json.loads(workflow_data)

#load prompt and data from file
with open("reprompt.json", "r", encoding="utf-8") as f:
    prompt_data = f.read()

reprompt = json.loads(prompt_data)


REPEAT = int(sys.argv[1])
#random seed
import random
for prompt_id, prompt_framework in reprompt.items():
    if prompt_id == "1":
        for i in range(REPEAT):
            image_path = os.path.join(prompt_id, os.path.join("base", "image.png"))
            for concept, instructions in prompt_framework['prompt_layers'].items():
                print(f"Processing prompt {i} for {concept}")
                # if concept != "dog6":
                #     continue
                seed = random.randint(1, 1000000000)
                #set the seed for our KSampler node
                workflow["11"]["inputs"]["seed"] = seed

                lora_name = lora_paths[concept][0]
                model_strength = lora_paths[concept][1]
                clip_strength = lora_paths[concept][2]
                workflow["39"]["inputs"]["lora_name"] = lora_name
                workflow["39"]["inputs"]["strength_model"] = model_strength
                workflow["39"]["inputs"]["strength_clip"] = clip_strength

                prompt = instructions["prompt"]
                workflow["12"]["inputs"]["text"] = prompt

                detection = instructions["detect"]
                workflow["19"]["inputs"]["text_input"] = detection
                #upload an image
                with open(image_path, "rb") as f:
                    comfyui_path_image = upload_file(f,"",True)

                #set the image name for our LoadImage node
                workflow["40"]["inputs"]["image"] = comfyui_path_image

                # #set model
                # workflow["14"]["inputs"]["ckpt_name"] = "meinamix_meinaV11.safetensors"

                ws = websocket.WebSocket()
                ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))
                images = get_images(ws, workflow)

                #Commented out code to display the output images:

                for node_id in images:
                    for image_data in images[node_id]:
                        from PIL import Image
                        import io
                        image = Image.open(io.BytesIO(image_data))
                        #image.show()
                        print(f"Saving image {i} for {concept}")
                        # save image
                        save_dir = os.path.join(prompt_id, concept)
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)
                        image_path = os.path.join(save_dir, f"{i}.png")
                        image.save(image_path)