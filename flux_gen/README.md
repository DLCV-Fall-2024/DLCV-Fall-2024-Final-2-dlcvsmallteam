# StageLight: Sequentially Targeted and Guided Editing for Layered Inpainting and Generative Hierarchical Transitions

This is the DLCV final project for multi-concept personalization image generation.

## Requirements
1. Download [ComfyUI](https://github.com/comfyanonymous/ComfyUI.git), follow the setup steps there. 
2. Download the below custom nodes, put them into the `ComfyUI/custom_nodes/` folder:
 - [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager.git) 
 - [ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack.git)
 - [ComfyUI-Impact-Subpack](https://github.com/ltdrdata/ComfyUI-Impact-Subpack.git)
 - [ComfyUI's ControlNet Auxiliary Preprocessors](https://github.com/Fannovel16/comfyui_controlnet_aux.git)
 - [WAS Node Suite](https://github.com/Fannovel16/comfyui_controlnet_aux.git)
 - [comfyui-art-venture](https://github.com/sipherxyz/comfyui-art-venture.git)
 - [ComfyUI-Advanced-ControlNet](https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet.git)
 - [rgthree's ComfyUI Nodes](https://github.com/rgthree/rgthree-comfy.git)
 - [KJNodes for COmfyUI](https://github.com/kijai/ComfyUI-KJNodes.git)
 - [ComfyUI-Florence2](https://github.com/kijai/ComfyUI-Florence2.git)
 - [ComfyUI-segment-anything-2](https://github.com/kijai/ComfyUI-segment-anything-2.git)
 - [ComfyUI-Easy-Use](https://github.com/yolain/ComfyUI-Easy-Use.git)
 - [Crystools](https://github.com/crystian/ComfyUI-Crystools.git)
 - [ComfyUI-iTools](https://github.com/MohammadAboulEla/ComfyUI-iTools.git)
 - [ControlAltAI-Nodes](https://github.com/gseth/ControlAltAI-Nodes.git)

3. Download the following model weights:
 - [Flux-dev-fp8](https://huggingface.co/XLabs-AI/flux-dev-fp8/tree/main), put into the `ComfyUI/models/diffusion_models` folder.
 - Download the Clips and VAEs by following the [example](https://comfyanonymous.github.io/ComfyUI_examples/flux/).
 - [Flux-dev-upscaler](https://huggingface.co/jasperai/Flux.1-dev-Controlnet-Upscaler/tree/main), put into the `ComfyUI/models/controlnet` folder.
 - [Flux-union-pro](https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro/tree/main), put into the `ComfyUI/models/controlnet` folder.

4. Put the loras in the `concepts` folder into the `ComfyUI/models/loras` folder, do not include the `concepts` folder or there will be an error.

## Run

```bash
bash run.sh $1 $2 $3
```
$1 is the absolute path to the ComfyUI folder, and $2 is the absolute path to this folder, and $3 is an integer of how many images you want for each prompt.



