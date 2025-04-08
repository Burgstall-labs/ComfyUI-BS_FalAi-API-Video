import os
import io
import base64
import uuid
import json
import time
import mimetypes
import traceback
import requests
import numpy as np
import torch
import scipy.io.wavfile
import fal_client # Assuming fal_client is installed: pip install fal
from PIL import Image
import cv2 # Requires opencv-python: pip install opencv-python

# Helper to access ComfyUI's path functions
import folder_paths

# --- Configuration Data with Categories ---
MODEL_CONFIGS = {
    "image_to_video": {
        "MiniMax (Hailuo AI) Video 01 Image to Video": {
            "endpoint": "fal-ai/minimax/video-01/image-to-video",
            "resolutions": [], # Not specified, assume width/height
            "aspect_ratios": [], # Not specified
            "durations": [], # Not specified
            "schema_str": "[Prompt:String], [Image_url:String]",
        },
        "Kling 1.6 Image to Video (Pro)": { # Added (Pro) for clarity
            "endpoint": "fal-ai/kling-video/v1.6/pro/image-to-video",
            "resolutions": [], # Not specified, assume width/height used by Kling
            "aspect_ratios": ["16:9", "9:16", "1:1"],
            "durations": [5, 10], # Represented as ints
            "schema_str": "[Prompt:String], [Image_url:String], [duration:DurationEnum], [aspect_ratio:AspectRatioEnum]",
        },
        "Veo 2 (Image to Video) Image to Video": {
            "endpoint": "fal-ai/veo2/image-to-video",
            "resolutions": [], # Not specified
            "aspect_ratios": ["auto", "auto_prefer_portrait", "16:9", "9:16"],
            "durations": [5, 6, 7, 8], # Represented as ints (assuming 's' means seconds)
            "schema_str": "[Prompt:String], [Image_url:String], [aspect_ratio:AspectRatioEnum], [duration:DurationEnum]",
        },
        "PixVerse v4: Image to Video Fast Image to Video": {
            "endpoint": "fal-ai/pixverse/v4/image-to-video/fast",
            "resolutions": ["360p", "540p", "720p"],
            "aspect_ratios": ["16:9", "4:3", "1:1", "3:4", "9:16"],
            "durations": [], # Not specified
            "schema_str": "[Prompt:String], [aspect_ratio:AspectRatioEnum], [resolution:ResolutionEnum], [negative_prompt:String], [style:Enum], [seed:Integer], [image_url:String]",
        },
        "PixVerse v4: Image to Video Image to Video": {
            "endpoint": "fal-ai/pixverse/v4/image-to-video",
            "resolutions": ["360p", "540p", "720p", "1080p"],
            "aspect_ratios": ["16:9", "4:3", "1:1", "3:4", "9:16"],
            "durations": [5, 8], # Represented as ints
            "schema_str": "[Prompt:String], [aspect_ratio:AspectRatioEnum], [resolution:ResolutionEnum], [duration:DurationEnum], [negative_prompt:String], [style:Enum], [seed:Integer], [image_url:String]",
        },
        "Luma Ray 2 Flash (Image to Video) Image to Video": {
            "endpoint": "fal-ai/luma-dream-machine/ray-2-flash/image-to-video",
            "resolutions": ["540p", "720p", "1080p"],
            "aspect_ratios": ["16:9", "9:16", "4:3", "3:4", "21:9", "9:21"],
            "durations": [5], # Represented as ints
            "schema_str": "[Prompt:String], [image_url:String], [end_image_url:String], [aspect_ratio:AspectRatioEnum], [loop:Boolean], [resolution:ResolutionEnum], [duration:DurationEnum]",
        },
        "Pika Image to Video Turbo (v2) Image to Video": {
            "endpoint": "fal-ai/pika/v2/turbo/image-to-video",
            "resolutions": ["720p", "1080p"],
            "aspect_ratios": ["16:9", "9:16", "1:1", "4:5", "5:4", "3:2", "2:3"],
            "durations": [], # Specified via duration:Integer in schema
            "schema_str": "[Image_url:String], [Prompt:String], [seed:Integer], [negative_prompt:String], [resolution:ResolutionEnum], [duration:Integer]",
        },
        "Wan-2.1 Image-to-Video Image to Video": {
            "endpoint": "fal-ai/wan-i2v", # Simplified endpoint name if this works
            "resolutions": ["480p", "720p"],
            "aspect_ratios": ["auto", "16:9", "9:16", "1:1"],
            "durations": [5], # Represented as ints
            "schema_str": "[Prompt:String], [negative_prompt:String], [image_url:String], [num_frames:Integer], [frames_per_second:Integer], [seed:Integer], [motion:Integer], [resolution:ResolutionEnum], [num_inference_steps:Integer]",
        },
        "MiniMax (Hailuo AI) Video 01 Director - Image to Video Image to Video": {
            "endpoint": "fal-ai/minimax/video-01-director/image-to-video",
            "resolutions": [], # Not specified
            "aspect_ratios": ["16:9", "9:16", "1:1"],
            "durations": [], # Not specified
            "schema_str": "[Prompt:String], [Image_url:String], [prompt_optimizer:Boolean]",
        },
         "Skyreels V1 (Image-to-Video) Image to Video": {
            "endpoint": "fal-ai/skyreels-i2v", # Simplified endpoint name if this works
            "resolutions": [], # Not specified
            "aspect_ratios": ["16:9", "9:16", "1:1"],
            "durations": [], # Not specified
            "schema_str": "[Prompt:String], [Image_url:String], [seed:Integer], [guidance_scale:Float], [num_inference_steps:Integer], [negative_prompt:String], [aspect_ratio:AspectRatioEnum]",
        },
        "Kling 1.6 Image to Video (Standard)": { # Added (Standard) for clarity
            "endpoint": "fal-ai/kling-video/v1.6/standard/image-to-video",
            "resolutions": [], # Not specified
            "aspect_ratios": ["16:9", "9:16", "1:1"],
            "durations": [5, 10],
            "schema_str": "[Prompt:String], [Image_url:String], [duration:DurationEnum], [aspect_ratio:AspectRatioEnum]",
        },
         "MiniMax (Hailuo AI) Video 01 Live Image to Video": {
            "endpoint": "fal-ai/minimax/video-01-live/image-to-video",
            "resolutions": [], # Not specified
            "aspect_ratios": [], # Not specified
            "durations": [], # Not specified
            "schema_str": "[Prompt:String], [Image_url:String], [prompt_optimizer:Boolean]",
         },
         "Kling 1.5 Image to Video (Pro)": { # Added (Pro) for clarity
            "endpoint": "fal-ai/kling-video/v1.5/pro/image-to-video",
            "resolutions": [], # Not specified
            "aspect_ratios": ["16:9", "9:16", "1:1"],
            "durations": [5, 10],
            "schema_str": "[Prompt:String], [Image_url:String], [duration:DurationEnum], [aspect_ratio:AspectRatioEnum]",
        },
        "Pika Image to Video (v2.2) Image to Video": {
            "endpoint": "fal-ai/pika/v2.2/image-to-video",
            "resolutions": ["720p", "1080p"],
            "aspect_ratios": ["16:9", "9:16", "1:1", "4:5", "5:4", "3:2", "2:3"],
            "durations": [],  # Specified via duration:Integer in schema
            "schema_str": "[Image_url:String], [Prompt:String], [seed:Integer], [negative_prompt:String], [resolution:ResolutionEnum], [duration:Integer]"
        },
        "Pika Image to Video (v2.1) Image to Video": {
            "endpoint": "fal-ai/pika/v2.1/image-to-video",
            "resolutions": ["720p", "1080p"],
            "aspect_ratios": ["16:9", "9:16", "1:1", "4:5", "5:4", "3:2", "2:3"],
            "durations": [],  # Specified via duration:Integer in schema
            "schema_str": "[Image_url:String], [Prompt:String], [seed:Integer], [negative_prompt:String], [resolution:ResolutionEnum], [duration:Integer]"
        },
        "Vidu Image to Video": {
            "endpoint": "fal-ai/vidu/image-to-video",
            "resolutions": [],
            "aspect_ratios": ["16:9", "9:16", "1:1"],
            "durations": [3],
            "schema_str": "[image_url:String], [prompt:String], [negative_prompt:String], [seed:Integer], [num_frames:Integer], [fps:Integer], [guidance_scale:Float], [num_inference_steps:Integer]"
        },
         "WAN Pro Image to Video": {
            "endpoint": "fal-ai/wan-pro/image-to-video",
            "resolutions": ["480p", "720p"],
            "aspect_ratios": ["auto", "16:9", "9:16", "1:1"],
            "durations": [5],
            "schema_str": "[Prompt:String], [negative_prompt:String], [image_url:String], [num_frames:Integer], [frames_per_second:Integer], [seed:Integer], [motion:Integer], [resolution:ResolutionEnum], [num_inference_steps:Integer]"
        },
        "Hunyuan Video (Image to Video)": {
            "endpoint": "fal-ai/hunyuan-video-image-to-video",
            "resolutions": ["256p", "512p"],
            "aspect_ratios": ["1:1"],
            "durations": [],
            "schema_str": "[Prompt:String], [image_url:String], [seed:Integer], [negative_prompt:String]"
        },
        "LTX Video v0.95 Image to Video": {
            "endpoint": "fal-ai/ltx-video-v095/image-to-video",
            "resolutions": ["256p", "512p"],
            "aspect_ratios": ["16:9", "9:16", "1:1"],
            "durations": [2,3,4,5,6,7,8,9,10],
            "schema_str": "[Prompt:String], [image_url:String], [negative_prompt:String], [seed:Integer], [num_frames:Integer], [fps:Integer], [guidance_scale:Float], [num_inference_steps:Integer]"
        },
          "Luma Dream Machine (Image to Video) Image to Video": {
            "endpoint": "fal-ai/luma-dream-machine/image-to-video",
            "resolutions": ["540p", "720p", "1080p"],
            "aspect_ratios": ["16:9", "9:16", "4:3", "3:4", "21:9", "9:21"],
            "durations": [5],
            "schema_str": "[Prompt:String], [image_url:String], [end_image_url:String], [aspect_ratio:AspectRatioEnum], [loop:Boolean], [resolution:ResolutionEnum], [duration:DurationEnum]"
        },
        "Luma Ray 2 (Image to Video) Image to Video": {
            "endpoint": "fal-ai/luma-dream-machine/ray-2/image-to-video",
            "resolutions": ["540p", "720p", "1080p"],
            "aspect_ratios": ["16:9", "9:16", "4:3", "3:4", "21:9", "9:21"],
            "durations": [5],
            "schema_str": "[Prompt:String], [image_url:String], [end_image_url:String], [aspect_ratio:AspectRatioEnum], [loop:Boolean], [resolution:ResolutionEnum], [duration:DurationEnum]"
        },
        "Hunyuan Video (Image to Video - LoRA)": {
            "endpoint": "fal-ai/hunyuan-video-img2vid-lora",
            "resolutions": ["256p", "512p"],
            "aspect_ratios": ["1:1"],
            "durations": [],
            "schema_str": "[Prompt:String], [image_url:String], [seed:Integer], [negative_prompt:String]"
        },
        "PixVerse v3.5: Image to Video Image to Video": {
            "endpoint": "fal-ai/pixverse/v3.5/image-to-video",
            "resolutions": ["360p", "540p", "720p", "1080p"],
            "aspect_ratios": ["16:9", "4:3", "1:1", "3:4", "9:16"],
            "durations": [5, 8],
            "schema_str": "[Prompt:String], [aspect_ratio:AspectRatioEnum], [resolution:ResolutionEnum], [duration:DurationEnum], [negative_prompt:String], [style:Enum], [seed:Integer], [image_url:String]"
        },
        "PixVerse v3.5: Image to Video Fast Image to Video": {
            "endpoint": "fal-ai/pixverse/v3.5/image-to-video/fast",
            "resolutions": ["360p", "540p", "720p"],
            "aspect_ratios": ["16:9", "4:3", "1:1", "3:4", "9:16"],
            "durations": [],
            "schema_str": "[Prompt:String], [aspect_ratio:AspectRatioEnum], [resolution:ResolutionEnum], [negative_prompt:String], [style:Enum], [seed:Integer], [image_url:String]"
        },
         "LTX Video Image to Video": {
            "endpoint": "fal-ai/ltx-video/image-to-video",
            "resolutions": ["256p", "512p"],
            "aspect_ratios": ["16:9", "9:16", "1:1"],
            "durations": [2,3,4,5,6,7,8,9,10],
            "schema_str": "[Prompt:String], [image_url:String], [negative_prompt:String], [seed:Integer], [num_frames:Integer], [fps:Integer], [guidance_scale:Float], [num_inference_steps:Integer]"
        },
        "CogVideoX 5B Image to Video": {
            "endpoint": "fal-ai/cogvideox-5b/image-to-video",
            "resolutions": [],
            "aspect_ratios": ["16:9", "9:16", "1:1"],
            "durations": [2,3,4,5,6,7,8,9,10],
            "schema_str": "[Prompt:String], [image_url:String], [seed:Integer], [negative_prompt:String], [num_frames:Integer], [fps:Integer], [guidance_scale:Float], [num_inference_steps:Integer]"
        },
          "Kling 1.5 Image to Video (Pro)": { # Added (Pro) for clarity
            "endpoint": "fal-ai/kling-video/v1.5/pro/image-to-video",
            "resolutions": [], # Not specified
            "aspect_ratios": ["16:9", "9:16", "1:1"],
            "durations": [5, 10],
            "schema_str": "[Prompt:String], [Image_url:String], [duration:DurationEnum], [aspect_ratio:AspectRatioEnum]",
        },
          "Kling 1 Image to Video (Pro)": { # Added (Pro) for clarity
            "endpoint": "fal-ai/kling-video/v1/pro/image-to-video",
            "resolutions": [], # Not specified
            "aspect_ratios": ["16:9", "9:16", "1:1"],
            "durations": [5, 10],
            "schema_str": "[Prompt:String], [Image_url:String], [duration:DurationEnum], [aspect_ratio:AspectRatioEnum]",
        },
         "Kling 1 Image to Video (Standard)": { # Added (Pro) for clarity
            "endpoint": "fal-ai/kling-video/v1/standard/image-to-video",
            "resolutions": [], # Not specified
            "aspect_ratios": ["16:9", "9:16", "1:1"],
            "durations": [5, 10],
            "schema_str": "[Prompt:String], [Image_url:String], [duration:DurationEnum], [aspect_ratio:AspectRatioEnum]",
        },
    },
    "text_to_video": {
        "Kling 1.6 Text to Video (Pro)": { # Moved here
            "endpoint": "fal-ai/kling-video/v1.6/pro/text-to-video",
            "resolutions": [],
            "aspect_ratios": ["16:9", "9:16", "1:1"],
            "durations": [5, 10],
            "schema_str": "[Prompt:String], [negative_prompt:String], [duration:DurationEnum], [aspect_ratio:AspectRatioEnum], [seed:Integer], [width:Integer], [height:Integer], [motion_bucket_id:Integer], [cond_aug:Float], [steps:Integer], [guidance_scale:Float], [fps:Integer]",
        },
        "Pika Text to Video": {
            "endpoint": "fal-ai/pika/v2/text-to-video",
            "resolutions": ["720p", "1024p"],
            "aspect_ratios": ["16:9", "9:16", "1:1", "4:5", "5:4", "3:2", "2:3"],
            "durations": [], # Specified via duration:Integer in schema
            "schema_str": "[prompt:String], [negative_prompt:String], [seed:Integer], [resolution:ResolutionEnum], [duration:Integer]"
        },
        "Luma Dream Machine Text to Video": {
            "endpoint": "fal-ai/luma-dream-machine/text-to-video",
            "resolutions": ["540p", "720p", "1080p"],
            "aspect_ratios": ["16:9", "9:16", "4:3", "3:4", "21:9", "9:21"],
            "durations": [5],
            "schema_str": "[prompt:String], [seed:Integer], [aspect_ratio:AspectRatioEnum], [loop:Boolean], [resolution:ResolutionEnum], [duration:DurationEnum]"
        },
        "PixVerse v4 Text to Video": {
            "endpoint": "fal-ai/pixverse/v4/text-to-video",
            "resolutions": ["360p", "540p", "720p", "1080p"],
            "aspect_ratios": ["16:9", "4:3", "1:1", "3:4", "9:16"],
            "durations": [5, 8],
            "schema_str": "[prompt:String], [negative_prompt:String], [style:Enum], [seed:Integer], [aspect_ratio:AspectRatioEnum], [resolution:ResolutionEnum], [duration:DurationEnum]"
        },
        "MiniMax (Hailuo AI) Video 01 Text to Video": {
            "endpoint": "fal-ai/minimax/video-01/text-to-video",
            "resolutions": [], # Unknown
            "aspect_ratios": [],
            "durations": [],
            "schema_str": "[prompt:String]"
        },
        "Hunyuan Video Text to Video": {
            "endpoint": "fal-ai/hunyuan-video",
            "resolutions": ["256p", "512p"],
            "aspect_ratios": ["1:1"],
            "durations": [],
            "schema_str": "[prompt:String], [seed:Integer], [negative_prompt:String]"
        },
    },
    # --- Add other types in the future ---
}

# --- Helper Function to Parse Schema String (Keep as is) ---
def parse_schema(schema_str):
    """Parses the simple schema string into a set of expected parameter names."""
    params = set()
    # Remove brackets and split
    parts = schema_str.strip('[]').split('], [')
    for part in parts:
        if ':' in part:
            param_name = part.split(':')[0].strip()
            # Normalize common variations
            if param_name == "Image_url": param_name = "image_url"
            if param_name == "Prompt": param_name = "prompt"
            # Map schema names to input names if they differ
            if param_name == "duration": param_name = "duration_seconds" # Match input name
            if param_name == "num_inference_steps": param_name = "steps" # Match input name
            if param_name == "resolution": param_name = "resolution_enum" # Match input name
            if param_name == "aspect_ratio": param_name = "aspect_ratio_enum" # Match input name
            # Add other mappings if needed (e.g., style, motion, etc.)
            params.add(param_name)
    return params

# --- Populate Model Configs with Parsed Schema (Handles nested structure) ---
for category, models in MODEL_CONFIGS.items():
    for name, config in models.items():
        config['expected_params'] = parse_schema(config['schema_str'])

# --- Dynamically Create Dropdown Lists (Combined from all categories) ---
ALL_MODEL_NAMES_I2V = sorted(list(MODEL_CONFIGS["image_to_video"].keys()))
ALL_MODEL_NAMES_T2V = sorted(list(MODEL_CONFIGS["text_to_video"].keys()))

# Combine options from both categories for general UI elements
ALL_RESOLUTIONS = sorted(list(set(res for category in MODEL_CONFIGS.values() for cfg in category.values() for res in cfg['resolutions'])))
ALL_ASPECT_RATIOS = sorted(list(set(ar for category in MODEL_CONFIGS.values() for cfg in category.values() for ar in cfg['aspect_ratios'])))

# Add common defaults if lists are empty
if not ALL_RESOLUTIONS: ALL_RESOLUTIONS = ["720p", "1080p", "512p", "576p"] # Add common ones
if not ALL_ASPECT_RATIOS: ALL_ASPECT_RATIOS = ["16:9", "9:16", "1:1", "4:3", "3:4"]

# Add a "None" or "Auto" option where appropriate
if "auto" not in ALL_ASPECT_RATIOS: ALL_ASPECT_RATIOS.insert(0, "auto")
ALL_RESOLUTIONS.insert(0, "auto") # Use 'auto' as default/unspecified


# --- Define the Image-to-Video Node Class ---
class FalAPIVideoGeneratorI2V:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (ALL_MODEL_NAMES_I2V,), # Use I2V models
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "Paste FAL_KEY credentials here (e.g., key_id:key_secret)"
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "prompt": ("STRING", {"multiline": True, "default": "A wild Burgstall appears"}),
            },
            "optional": {
                "image": ("IMAGE",), # Keep image input for I2V
                "negative_prompt": ("STRING", {"multiline": True, "default": "Ugly, blurred, distorted"}),
                "resolution_enum": (ALL_RESOLUTIONS, {"default": "auto"}),
                "aspect_ratio_enum": (ALL_ASPECT_RATIOS, {"default": "auto"}),
                "duration_seconds": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 30.0, "step": 0.5}),
                "guidance_scale": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 100, "step": 1}),
                "num_frames": ("INT", {"default": 0, "min": 0, "max": 1000}),
                "style": (["auto", "cinematic", "anime", "photorealistic", "fantasy", "cartoon"], {"default": "auto"}),
                "prompt_optimizer": ("BOOLEAN", {"default": False}),
                "cleanup_temp_video": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_batch",)
    FUNCTION = "generate_video"
    CATEGORY = "BS_FalAi-API-Video/Image-to-Video" # Specific category

    # --- _prepare_image method (Needed for I2V) ---
    def _prepare_image(self, image_tensor, target_width=None, target_height=None):
        """Converts ComfyUI Image Tensor to Base64 Data URI, resizing if needed."""
        if image_tensor is None:
            print("FalAPIVideoGeneratorI2V: No image provided to _prepare_image.")
            return None

        print("FalAPIVideoGeneratorI2V: Preparing image...")
        try:
            if image_tensor.dim() == 4 and image_tensor.shape[0] == 1:
                img_tensor = image_tensor[0]
            elif image_tensor.dim() == 3:
                img_tensor = image_tensor
            else:
                raise ValueError(f"Unexpected image tensor shape: {image_tensor.shape}")

            img_np = img_tensor.cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            pil_image = Image.fromarray(img_np)

            # Optional Resizing (unlikely to be used now without width/height inputs)
            if target_width and target_height and \
               (pil_image.width != target_width or pil_image.height != target_height):
                print(f"FalAPIVideoGeneratorI2V: Resizing input image from {pil_image.width}x{pil_image.height} to {target_width}x{target_height}")
                pil_image = pil_image.resize((target_width, target_height), Image.LANCZOS)

            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG")
            img_bytes = buffered.getvalue()
            img_b64_encoded = base64.b64encode(img_bytes).decode('utf-8')
            img_data_uri = f"data:image/png;base64,{img_b64_encoded}"
            print("FalAPIVideoGeneratorI2V: Image preparation complete.")
            return img_data_uri

        except Exception as e:
            print(f"ERROR: FalAPIVideoGeneratorI2V: Image processing failed: {e}")
            traceback.print_exc()
            return None

    # --- generate_video method for I2V ---
    def generate_video(self, model_name, api_key, seed, prompt,
                         image=None, negative_prompt=None, # Keep image param
                         resolution_enum="auto", aspect_ratio_enum="auto",
                         duration_seconds=5.0,
                         guidance_scale=7.5, steps=25,
                         num_frames=0,
                         style="auto",
                         prompt_optimizer=False, cleanup_temp_video=True):

        # --- 1. Get Model Config (from image_to_video category) ---
        if model_name not in MODEL_CONFIGS["image_to_video"]:
            print(f"ERROR: FalAPIVideoGeneratorI2V: Unknown model name '{model_name}' in image_to_video category.")
            return (None,)
        config = MODEL_CONFIGS["image_to_video"][model_name] # Use correct category
        endpoint_id = config['endpoint']
        expected_params = config['expected_params']
        print(f"FalAPIVideoGeneratorI2V: Selected Model: {model_name}")
        print(f"FalAPIVideoGeneratorI2V: Endpoint: {endpoint_id}")

        # --- 2. API Key Setup (Same as before) ---
        if not api_key or not api_key.strip() or api_key == "Paste FAL_KEY credentials here (e.g., key_id:key_secret)":
            print("ERROR: FalAPIVideoGeneratorI2V: API Key input field is empty or contains default text. Please paste your actual FAL key.")
            return (None,)
        api_key_value = api_key.strip()
        if ':' not in api_key_value and len(api_key_value) < 20:
             print("WARN: FalAPIVideoGeneratorI2V: API Key doesn't seem to follow the usual 'key_id:key_secret' format. Ensure it's correct.")
        try:
            os.environ["FAL_KEY"] = api_key_value
            print("FalAPIVideoGeneratorI2V: Using provided API Key.")
        except Exception as e:
             print(f"ERROR: FalAPIVideoGeneratorI2V: Failed to set API Key environment variable: {e}")
             traceback.print_exc(); return (None,)

        # --- 3. Prepare Payload Dynamically ---
        payload = {}
        img_data_uri = None

        # --- Image Handling (Keep for I2V) ---
        if 'image_url' in expected_params:
            if image is not None:
                img_data_uri = self._prepare_image(image, target_width=None, target_height=None)
                if img_data_uri:
                    payload['image_url'] = img_data_uri
                else:
                    print("ERROR: FalAPIVideoGeneratorI2V: Failed to prepare image, aborting.")
                    return (None,)
            else:
                print(f"WARN: FalAPIVideoGeneratorI2V: Model '{model_name}' expects 'image_url', but no image was provided.")
                # Consider making this an error: return (None,)
        elif image is not None:
             print(f"WARN: FalAPIVideoGeneratorI2V: Image provided, but model '{model_name}' does not seem to expect 'image_url'. Image ignored.")

        # --- Text Prompts (Same as before) ---
        if 'prompt' in expected_params:
            if prompt and prompt.strip():
                 payload['prompt'] = prompt.strip()
            else:
                 print(f"WARN: FalAPIVideoGeneratorI2V: Model '{model_name}' expects 'prompt', but it's empty.")
        if 'negative_prompt' in expected_params and negative_prompt and negative_prompt.strip():
            payload['negative_prompt'] = negative_prompt.strip()

        # --- Numeric/Boolean/Enum Parameters (Same logic) ---
        param_map = {
            "seed": seed,
            "duration_seconds": duration_seconds,
            "guidance_scale": guidance_scale,
            "steps": steps,
            "num_frames": num_frames,
            "style": style,
            "prompt_optimizer": prompt_optimizer,
            "resolution_enum": resolution_enum,
            "aspect_ratio_enum": aspect_ratio_enum
        }

        api_name_map = {
            "duration_seconds": "duration",
            "steps": "num_inference_steps",
            "resolution_enum": "resolution",
            "aspect_ratio_enum": "aspect_ratio"
        }

        for input_name, value in param_map.items():
            api_name = api_name_map.get(input_name, input_name) # Default to input name if no mapping
            if api_name in expected_params:
                if isinstance(value, str) and value.lower() == "auto":
                    print(f"FalAPIVideoGeneratorI2V: Skipping '{api_name}' parameter with value 'auto'.")
                    continue
                if isinstance(value, (int, float)) and value == 0 and input_name in ["num_frames", "seed"]: # Skip default 0s where appropriate
                    if input_name == "seed" and 0 in expected_params: # Only skip seed=0 if it's truly optional
                        print(f"FalAPIVideoGeneratorI2V: Skipping '{api_name}' parameter with default value {value}.")
                        continue
                    elif input_name == "num_frames":
                        print(f"FalAPIVideoGeneratorI2V: Skipping '{api_name}' parameter with default value {value}.")
                        continue

                # Handle specific type conversions if needed based on schema
                if api_name == "duration" and "duration:Integer" in config.get('schema_str', ''):
                     payload[api_name] = int(value)
                elif api_name == "num_inference_steps" and "num_inference_steps:Integer" in config.get('schema_str', ''):
                     payload[api_name] = int(value)
                else:
                    payload[api_name] = value


        # --- Handle width/height (via enums, same logic but context is I2V) ---
        used_enum_res = False
        if 'resolution' in expected_params and resolution_enum != "auto":
            payload['resolution'] = resolution_enum
            used_enum_res = True
            payload.pop('width', None); payload.pop('height', None)

        used_enum_ar = False
        if 'aspect_ratio' in expected_params and aspect_ratio_enum != "auto":
            payload['aspect_ratio'] = aspect_ratio_enum
            used_enum_ar = True
            payload.pop('width', None); payload.pop('height', None)

        if 'width' in expected_params and not used_enum_res and not used_enum_ar:
            print(f"WARN: FalAPIVideoGeneratorI2V: Model '{model_name}' expects 'width', but no explicit width/height inputs exist and resolution/aspect ratio enums are 'auto' or not used. API might use defaults or fail.")
        if 'height' in expected_params and not used_enum_res and not used_enum_ar:
            print(f"WARN: FalAPIVideoGeneratorI2V: Model '{model_name}' expects 'height', but no explicit width/height inputs exist and resolution/aspect ratio enums are 'auto' or not used. API might use defaults or fail.")


        # --- 4. API Call, Download, Frame Extraction (Same logic) ---
        request_id = None
        video_url = None
        temp_video_filepath = None

        try:
            print(f"FalAPIVideoGeneratorI2V: Submitting job to endpoint: {endpoint_id}")
            # print(f"FalAPIVideoGeneratorI2V: Payload: {payload}") # Debug
            if not endpoint_id or not endpoint_id.strip():
                 raise ValueError("Endpoint ID cannot be empty.")

            handler = fal_client.submit(endpoint_id.strip(), arguments=payload)
            request_id = handler.request_id
            print(f"FalAPIVideoGeneratorI2V: Job submitted. Request ID: {request_id}")
            print("FalAPIVideoGeneratorI2V: Waiting for Fal.ai job completion...")
            response = handler.get()
            print("FalAPIVideoGeneratorI2V: Fal.ai job completed.")

            # Process Response (Same as before)
            video_data = response.get('video')
            if isinstance(video_data, dict): video_url = video_data.get('url')
            elif isinstance(response.get('videos'), list) and len(response['videos']) > 0:
                 vid_info = response['videos'][0]
                 if isinstance(vid_info, dict): video_url = vid_info.get('url')
            elif isinstance(response, dict) and response.get('url') and isinstance(response.get('url'), str):
                 if any(ext in response['url'].lower() for ext in ['.mp4', '.webm', '.mov', '.avi']):
                      video_url = response['url']

            if not video_url:
                 print(f"ERROR: FalAPIVideoGeneratorI2V: Could not find video 'url' in Fal.ai response.")
                 print(f"Full response: {response}")
                 return (None,)

            print(f"FalAPIVideoGeneratorI2V: Video URL received: {video_url}")
            print("FalAPIVideoGeneratorI2V: Downloading video file...")
            video_response = requests.get(video_url, stream=True, timeout=300)
            video_response.raise_for_status()

            # Save to Temporary File (Same as before)
            output_dir = folder_paths.get_temp_directory()
            os.makedirs(output_dir, exist_ok=True)
            content_type = video_response.headers.get('content-type', '').lower()
            extension = '.mp4'
            if 'video/mp4' in content_type: extension = '.mp4'
            elif 'video/webm' in content_type: extension = '.webm'
            elif video_url.lower().endswith('.mp4'): extension = '.mp4'
            elif video_url.lower().endswith('.webm'): extension = '.webm'
            filename = f"fal_api_i2v_temp_{uuid.uuid4().hex}{extension}"
            temp_video_filepath = os.path.join(output_dir, filename)
            with open(temp_video_filepath, 'wb') as video_file:
                for chunk in video_response.iter_content(chunk_size=1024*1024): video_file.write(chunk)
            print(f"FalAPIVideoGeneratorI2V: Video downloaded to: {temp_video_filepath}")

            # Extract Frames (Same as before)
            print(f"FalAPIVideoGeneratorI2V: Extracting frames...")
            frames_list = []
            cap = cv2.VideoCapture(temp_video_filepath)
            if not cap.isOpened():
                if cleanup_temp_video and os.path.exists(temp_video_filepath):
                    try: os.remove(temp_video_filepath)
                    except Exception as clean_e: print(f"WARN: Failed cleanup: {clean_e}")
                raise IOError(f"Could not open video file: {temp_video_filepath}")
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret: break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_list.append(frame_rgb)
                frame_count += 1
            cap.release()
            if not frames_list:
                if cleanup_temp_video and os.path.exists(temp_video_filepath):
                     try: os.remove(temp_video_filepath)
                     except Exception as clean_e: print(f"WARN: Failed cleanup: {clean_e}")
                raise ValueError(f"No frames extracted from video: {temp_video_filepath}")
            print(f"FalAPIVideoGeneratorI2V: Extracted {frame_count} frames.")

            # Convert to Tensor (Same as before)
            frames_np = np.stack(frames_list, axis=0)
            frames_tensor = torch.from_numpy(frames_np).float() / 255.0
            print(f"FalAPIVideoGeneratorI2V: Frames tensor shape: {frames_tensor.shape}")
            return (frames_tensor,)

        # Exception Handling (Same as before, update class name in logs)
        except requests.exceptions.RequestException as e:
             url_for_error = video_url if video_url else f"API Endpoint: {endpoint_id}"
             print(f"ERROR: FalAPIVideoGeneratorI2V: Network request failed ({url_for_error}): {e}")
             traceback.print_exc(); return (None,)
        except (cv2.error, IOError, ValueError) as e:
             print(f"ERROR: FalAPIVideoGeneratorI2V: Video processing/Frame extraction error: {e}")
             traceback.print_exc(); return (None,)
        except Exception as e:
            req_id_str = request_id if request_id else 'N/A'
            print(f"ERROR: FalAPIVideoGeneratorI2V: Unexpected error (Request ID: {req_id_str}): {e}")
            if endpoint_id and ("not found" in str(e).lower() or "404" in str(e)):
                 print("--- Hint: Check if the Endpoint ID is correct and accessible. ---")
            traceback.print_exc(); return (None,)

        finally:
            # Cleanup (Same as before, update class name in logs)
            if cleanup_temp_video and temp_video_filepath and os.path.exists(temp_video_filepath):
                try:
                    print(f"FalAPIVideoGeneratorI2V: Cleaning up temporary video file: {temp_video_filepath}")
                    os.remove(temp_video_filepath)
                except Exception as e:
                    print(f"WARN: FalAPIVideoGeneratorI2V: Failed to delete temp file '{temp_video_filepath}': {e}")
            elif temp_video_filepath and os.path.exists(temp_video_filepath):
                 print(f"FalAPIVideoGeneratorI2V: Keeping temporary video file: {temp_video_filepath}")


# --- Define the Text-to-Video Node Class ---
class FalAPIVideoGeneratorT2V:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (ALL_MODEL_NAMES_T2V,), # Use T2V models
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "Paste FAL_KEY credentials here (e.g., key_id:key_secret)"
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "prompt": ("STRING", {"multiline": True, "default": "A wild Burgstall appears"}), # Changed default prompt
            },
            "optional": {
                # REMOVED "image": ("IMAGE",),
                "negative_prompt": ("STRING", {"multiline": True, "default": "Blurry, low quality, text, watermark"}), # Changed default negative
                "resolution_enum": (ALL_RESOLUTIONS, {"default": "auto"}),
                "aspect_ratio_enum": (ALL_ASPECT_RATIOS, {"default": "auto"}),
                "duration_seconds": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 30.0, "step": 0.5}),
                "guidance_scale": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 100, "step": 1}),
                "num_frames": ("INT", {"default": 0, "min": 0, "max": 1000}),
                "style": (["auto", "cinematic", "anime", "photorealistic", "fantasy", "cartoon"], {"default": "auto"}),
                "prompt_optimizer": ("BOOLEAN", {"default": False}),
                "cleanup_temp_video": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_batch",)
    FUNCTION = "generate_video"
    CATEGORY = "BS_FalAi-API-Video/Text-to-Video" # Specific category

    # NO _prepare_image method needed for T2V

    # --- generate_video method for T2V ---
    def generate_video(self, model_name, api_key, seed, prompt,
                         # REMOVED image=None parameter
                         negative_prompt=None,
                         resolution_enum="auto", aspect_ratio_enum="auto",
                         duration_seconds=5.0,
                         guidance_scale=7.5, steps=25,
                         num_frames=0,
                         style="auto",
                         prompt_optimizer=False, cleanup_temp_video=True):

        # --- 1. Get Model Config (from text_to_video category) ---
        if model_name not in MODEL_CONFIGS["text_to_video"]:
            print(f"ERROR: FalAPIVideoGeneratorT2V: Unknown model name '{model_name}' in text_to_video category.")
            return (None,)
        config = MODEL_CONFIGS["text_to_video"][model_name] # Use correct category
        endpoint_id = config['endpoint']
        expected_params = config['expected_params']
        print(f"FalAPIVideoGeneratorT2V: Selected Model: {model_name}")
        print(f"FalAPIVideoGeneratorT2V: Endpoint: {endpoint_id}")

        # --- 2. API Key Setup (Same as before) ---
        if not api_key or not api_key.strip() or api_key == "Paste FAL_KEY credentials here (e.g., key_id:key_secret)":
            print("ERROR: FalAPIVideoGeneratorT2V: API Key input field is empty or contains default text. Please paste your actual FAL key.")
            return (None,)
        api_key_value = api_key.strip()
        if ':' not in api_key_value and len(api_key_value) < 20:
             print("WARN: FalAPIVideoGeneratorT2V: API Key doesn't seem to follow the usual 'key_id:key_secret' format. Ensure it's correct.")
        try:
            os.environ["FAL_KEY"] = api_key_value
            print("FalAPIVideoGeneratorT2V: Using provided API Key.")
        except Exception as e:
             print(f"ERROR: FalAPIVideoGeneratorT2V: Failed to set API Key environment variable: {e}")
             traceback.print_exc(); return (None,)

        # --- 3. Prepare Payload Dynamically ---
        payload = {}
        # NO Image Handling section needed for T2V

        # --- Text Prompts (Same as before) ---
        if 'prompt' in expected_params:
            if prompt and prompt.strip():
                 payload['prompt'] = prompt.strip()
            else:
                 print(f"WARN: FalAPIVideoGeneratorT2V: Model '{model_name}' expects 'prompt', but it's empty.")
        if 'negative_prompt' in expected_params and negative_prompt and negative_prompt.strip():
            payload['negative_prompt'] = negative_prompt.strip()

        # --- Numeric/Boolean/Enum Parameters (Same logic) ---
        param_map = {
            "seed": seed,
            "duration_seconds": duration_seconds,
            "guidance_scale": guidance_scale,
            "steps": steps,
            "num_frames": num_frames,
            "style": style,
            "prompt_optimizer": prompt_optimizer,
            "resolution_enum": resolution_enum,
            "aspect_ratio_enum": aspect_ratio_enum
        }

        api_name_map = {
            "duration_seconds": "duration",
            "steps": "num_inference_steps",
            "resolution_enum": "resolution",
            "aspect_ratio_enum": "aspect_ratio"
        }

        for input_name, value in param_map.items():
            api_name = api_name_map.get(input_name, input_name)
            if api_name in expected_params:
                if isinstance(value, str) and value.lower() == "auto":
                    print(f"FalAPIVideoGeneratorT2V: Skipping '{api_name}' parameter with value 'auto'.")
                    continue
                if isinstance(value, (int, float)) and value == 0 and input_name in ["num_frames", "seed"]:
                    if input_name == "seed" and 0 in expected_params:
                        print(f"FalAPIVideoGeneratorT2V: Skipping '{api_name}' parameter with default value {value}.")
                        continue
                    elif input_name == "num_frames":
                        print(f"FalAPIVideoGeneratorT2V: Skipping '{api_name}' parameter with default value {value}.")
                        continue

                if api_name == "duration" and "duration:Integer" in config.get('schema_str', ''):
                     payload[api_name] = int(value)
                elif api_name == "num_inference_steps" and "num_inference_steps:Integer" in config.get('schema_str', ''):
                     payload[api_name] = int(value)
                else:
                    payload[api_name] = value


        # --- Handle width/height (via enums, same logic but context is T2V) ---
        used_enum_res = False
        if 'resolution' in expected_params and resolution_enum != "auto":
            payload['resolution'] = resolution_enum
            used_enum_res = True
            payload.pop('width', None); payload.pop('height', None)

        used_enum_ar = False
        if 'aspect_ratio' in expected_params and aspect_ratio_enum != "auto":
            payload['aspect_ratio'] = aspect_ratio_enum
            used_enum_ar = True
            payload.pop('width', None); payload.pop('height', None)

        if 'width' in expected_params and not used_enum_res and not used_enum_ar:
            print(f"WARN: FalAPIVideoGeneratorT2V: Model '{model_name}' expects 'width', but no explicit width/height inputs exist and resolution/aspect ratio enums are 'auto' or not used. API might use defaults or fail.")
        if 'height' in expected_params and not used_enum_res and not used_enum_ar:
            print(f"WARN: FalAPIVideoGeneratorT2V: Model '{model_name}' expects 'height', but no explicit width/height inputs exist and resolution/aspect ratio enums are 'auto' or not used. API might use defaults or fail.")


        # --- 4. API Call, Download, Frame Extraction (Same logic) ---
        request_id = None
        video_url = None
        temp_video_filepath = None

        try:
            print(f"FalAPIVideoGeneratorT2V: Submitting job to endpoint: {endpoint_id}")
            # print(f"FalAPIVideoGeneratorT2V: Payload: {payload}") # Debug
            if not endpoint_id or not endpoint_id.strip():
                 raise ValueError("Endpoint ID cannot be empty.")

            handler = fal_client.submit(endpoint_id.strip(), arguments=payload)
            request_id = handler.request_id
            print(f"FalAPIVideoGeneratorT2V: Job submitted. Request ID: {request_id}")
            print("FalAPIVideoGeneratorT2V: Waiting for Fal.ai job completion...")
            response = handler.get()
            print("FalAPIVideoGeneratorT2V: Fal.ai job completed.")

            # Process Response (Same as before)
            video_data = response.get('video')
            if isinstance(video_data, dict): video_url = video_data.get('url')
            elif isinstance(response.get('videos'), list) and len(response['videos']) > 0:
                 vid_info = response['videos'][0]
                 if isinstance(vid_info, dict): video_url = vid_info.get('url')
            elif isinstance(response, dict) and response.get('url') and isinstance(response.get('url'), str):
                 if any(ext in response['url'].lower() for ext in ['.mp4', '.webm', '.mov', '.avi']):
                      video_url = response['url']

            if not video_url:
                 print(f"ERROR: FalAPIVideoGeneratorT2V: Could not find video 'url' in Fal.ai response.")
                 print(f"Full response: {response}")
                 return (None,)

            print(f"FalAPIVideoGeneratorT2V: Video URL received: {video_url}")
            print("FalAPIVideoGeneratorT2V: Downloading video file...")
            video_response = requests.get(video_url, stream=True, timeout=300)
            video_response.raise_for_status()

            # Save to Temporary File (Same as before)
            output_dir = folder_paths.get_temp_directory()
            os.makedirs(output_dir, exist_ok=True)
            content_type = video_response.headers.get('content-type', '').lower()
            extension = '.mp4'
            if 'video/mp4' in content_type: extension = '.mp4'
            elif 'video/webm' in content_type: extension = '.webm'
            elif video_url.lower().endswith('.mp4'): extension = '.mp4'
            elif video_url.lower().endswith('.webm'): extension = '.webm'
            filename = f"fal_api_t2v_temp_{uuid.uuid4().hex}{extension}" # Different prefix
            temp_video_filepath = os.path.join(output_dir, filename)
            with open(temp_video_filepath, 'wb') as video_file:
                for chunk in video_response.iter_content(chunk_size=1024*1024): video_file.write(chunk)
            print(f"FalAPIVideoGeneratorT2V: Video downloaded to: {temp_video_filepath}")

            # Extract Frames (Same as before)
            print(f"FalAPIVideoGeneratorT2V: Extracting frames...")
            frames_list = []
            cap = cv2.VideoCapture(temp_video_filepath)
            if not cap.isOpened():
                if cleanup_temp_video and os.path.exists(temp_video_filepath):
                    try: os.remove(temp_video_filepath)
                    except Exception as clean_e: print(f"WARN: Failed cleanup: {clean_e}")
                raise IOError(f"Could not open video file: {temp_video_filepath}")
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret: break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_list.append(frame_rgb)
                frame_count += 1
            cap.release()
            if not frames_list:
                 if cleanup_temp_video and os.path.exists(temp_video_filepath):
                     try: os.remove(temp_video_filepath)
                     except Exception as clean_e: print(f"WARN: Failed cleanup: {clean_e}")
                 raise ValueError(f"No frames extracted from video: {temp_video_filepath}")
            print(f"FalAPIVideoGeneratorT2V: Extracted {frame_count} frames.")

            # Convert to Tensor (Same as before)
            frames_np = np.stack(frames_list, axis=0)
            frames_tensor = torch.from_numpy(frames_np).float() / 255.0
            print(f"FalAPIVideoGeneratorT2V: Frames tensor shape: {frames_tensor.shape}")
            return (frames_tensor,)

        # Exception Handling (Same as before, update class name in logs)
        except requests.exceptions.RequestException as e:
             url_for_error = video_url if video_url else f"API Endpoint: {endpoint_id}"
             print(f"ERROR: FalAPIVideoGeneratorT2V: Network request failed ({url_for_error}): {e}")
             traceback.print_exc(); return (None,)
        except (cv2.error, IOError, ValueError) as e:
             print(f"ERROR: FalAPIVideoGeneratorT2V: Video processing/Frame extraction error: {e}")
             traceback.print_exc(); return (None,)
        except Exception as e:
            req_id_str = request_id if request_id else 'N/A'
            print(f"ERROR: FalAPIVideoGeneratorT2V: Unexpected error (Request ID: {req_id_str}): {e}")
            if endpoint_id and ("not found" in str(e).lower() or "404" in str(e)):
                 print("--- Hint: Check if the Endpoint ID is correct and accessible. ---")
            traceback.print_exc(); return (None,)

        finally:
            # Cleanup (Same as before, update class name in logs)
            if cleanup_temp_video and temp_video_filepath and os.path.exists(temp_video_filepath):
                try:
                    print(f"FalAPIVideoGeneratorT2V: Cleaning up temporary video file: {temp_video_filepath}")
                    os.remove(temp_video_filepath)
                except Exception as e:
                    print(f"WARN: FalAPIVideoGeneratorT2V: Failed to delete temp file '{temp_video_filepath}': {e}")
            elif temp_video_filepath and os.path.exists(temp_video_filepath):
                 print(f"FalAPIVideoGeneratorT2V: Keeping temporary video file: {temp_video_filepath}")

def log_prefix():
    """Returns a standard prefix for log messages."""
    # You can customize this if you want different prefixes per node
    # return f"{__class__.__name__}:"
    return "FalAPIOmniProNode:" # Keep it specific for now

def _prepare_image_bytes(image_tensor):
    """Converts ComfyUI Image Tensor to PNG bytes."""
    if image_tensor is None:
        print(f"{log_prefix()} No image tensor provided to _prepare_image_bytes.")
        return None, None

    print(f"{log_prefix()} Preparing image tensor...")
    try:
        # Ensure tensor is on CPU and in the expected format (B, H, W, C) or (H, W, C)
        if image_tensor.dim() == 4 and image_tensor.shape[0] == 1:
            img_tensor = image_tensor.squeeze(0) # Remove batch dim
        elif image_tensor.dim() == 3:
            img_tensor = image_tensor
        else:
            raise ValueError(f"Unexpected image tensor shape: {image_tensor.shape}")

        img_tensor = img_tensor.cpu()

        # Convert to numpy array, scale if needed (assuming 0-1 float), convert type
        img_np = img_tensor.numpy()
        if img_np.max() <= 1.0 and img_np.min() >= 0.0:
             img_np = (img_np * 255)
        img_np = img_np.astype(np.uint8)

        # Create PIL Image
        pil_image = Image.fromarray(img_np, 'RGB') # Assume RGB

        # Save to bytes buffer
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        print(f"{log_prefix()} Image tensor preparation complete ({len(img_bytes)} bytes).")
        return img_bytes, "image/png"

    except Exception as e:
        print(f"ERROR: {log_prefix()} Image tensor processing failed: {e}")
        traceback.print_exc()
        return None, None

def _save_tensor_to_temp_video(image_tensor_batch, fps=30):
    """Saves a ComfyUI Image Tensor Batch (B, H, W, C) to a temporary MP4 file."""
    if image_tensor_batch is None or image_tensor_batch.dim() != 4 or image_tensor_batch.shape[0] == 0:
        print(f"{log_prefix()} Invalid or empty image tensor batch provided for video saving.")
        return None

    print(f"{log_prefix()} Saving image tensor batch to temporary video file...")
    batch_size, height, width, channels = image_tensor_batch.shape
    if channels != 3:
        print(f"ERROR: {log_prefix()} Expected 3 color channels (RGB) for video saving, got {channels}.")
        return None

    output_dir = folder_paths.get_temp_directory()
    os.makedirs(output_dir, exist_ok=True)
    filename = f"fal_omni_temp_upload_{uuid.uuid4().hex}.mp4"
    temp_video_filepath = os.path.join(output_dir, filename)

    # OpenCV expects BGR format and uint8 0-255 range
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for MP4
    video_writer = cv2.VideoWriter(temp_video_filepath, fourcc, float(fps), (width, height))

    if not video_writer.isOpened():
        print(f"ERROR: {log_prefix()} Failed to open video writer for path: {temp_video_filepath}")
        return None

    try:
        image_tensor_batch_cpu = image_tensor_batch.cpu() # Move to CPU once
        for i in range(batch_size):
            frame_tensor = image_tensor_batch_cpu[i]
            frame_np = frame_tensor.numpy()
            # Convert 0-1 float to 0-255 uint8
            if frame_np.max() <= 1.0 and frame_np.min() >= 0.0:
                frame_np = (frame_np * 255)
            frame_np = frame_np.astype(np.uint8)
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)

        print(f"{log_prefix()} Temporary video saved successfully to: {temp_video_filepath}")
        return temp_video_filepath

    except Exception as e:
        print(f"ERROR: {log_prefix()} Failed during video writing: {e}")
        traceback.print_exc()
        # Clean up partially written file if error occurs
        if video_writer.isOpened():
            video_writer.release()
        if os.path.exists(temp_video_filepath):
            try: os.remove(temp_video_filepath)
            except Exception as clean_e: print(f"WARN: {log_prefix()} Failed cleanup on error: {clean_e}")
        return None
    finally:
        if video_writer.isOpened():
            video_writer.release()

def _upload_media_to_fal(media_bytes, filename_hint, content_type):
    """
    Saves media bytes to a temporary file and uploads it using
    fal_client.upload_file, then returns the URL. Cleans up the temp file.
    """
    if not media_bytes:
        print(f"ERROR: {log_prefix()} No media bytes provided for upload ({filename_hint}).")
        return None

    temp_path = None # Initialize outside try block
    try:
        # Create a temporary file path
        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)
        # Try to use a reasonable extension from the hint or content_type
        ext = os.path.splitext(filename_hint)[1]
        if not ext and content_type:
            # Basic mapping from content type to extension
            if 'png' in content_type: ext = '.png'
            elif 'jpeg' in content_type or 'jpg' in content_type: ext = '.jpg'
            elif 'mp4' in content_type: ext = '.mp4'
            elif 'webm' in content_type: ext = '.webm'
            elif 'mp3' in content_type or 'mpeg' in content_type: ext = '.mp3'
            elif 'wav' in content_type: ext = '.wav'
            # Add more as needed
        if not ext: ext = ".tmp" # Fallback extension

        temp_filename = f"fal_upload_{uuid.uuid4().hex}{ext}"
        temp_path = os.path.join(temp_dir, temp_filename)

        # Write the bytes to the temporary file
        print(f"{log_prefix()} Writing temporary file for upload: {temp_path} ({len(media_bytes)} bytes)")
        with open(temp_path, "wb") as f:
            f.write(media_bytes)

        # Upload the temporary file using the correct function
        print(f"{log_prefix()} Uploading {temp_path} via fal_client.upload_file...")
        # VVVV THIS IS THE CORRECT FUNCTION CALL VVVV
        file_url = fal_client.upload_file(temp_path)
        # ^^^^ THIS IS THE CORRECT FUNCTION CALL ^^^^
        print(f"{log_prefix()} Upload successful. URL: {file_url}")
        return file_url

    except Exception as e:
        print(f"ERROR: {log_prefix()} Fal.ai media upload failed for {filename_hint} (using temp file {temp_path}): {e}")
        traceback.print_exc()
        return None # Indicate failure

    finally:
        # --- Crucial Cleanup ---
        # Ensure the temporary file is deleted whether upload succeeded or failed
        if temp_path and os.path.exists(temp_path):
            try:
                print(f"{log_prefix()} Cleaning up temporary upload file: {temp_path}")
                os.remove(temp_path)
            except Exception as cleanup_e:
                # Log cleanup error but don't mask the original error (if any)
                print(f"WARN: {log_prefix()} Failed to delete temporary upload file '{temp_path}': {cleanup_e}")


      
def _save_audio_tensor_to_temp_wav(audio_data):
    """
    Saves audio data (from ComfyUI AUDIO type) to a temporary WAV file.
    Returns the path to the temporary file.
    """
    if not audio_data or 'samples' not in audio_data or 'sample_rate' not in audio_data:
        print(f"ERROR: {log_prefix()} Invalid audio data received.")
        return None

    sample_rate = audio_data['sample_rate']
    samples_tensor = audio_data['samples']

    print(f"{log_prefix()} Processing audio tensor (Sample Rate: {sample_rate}, Shape: {samples_tensor.shape})")

    try:
        # Ensure tensor is on CPU
        samples_tensor = samples_tensor.cpu()

        # Handle potential batch dimension (use the first item if batched)
        if samples_tensor.dim() == 3: # (batch, channels, samples)
            print(f"{log_prefix()} Audio tensor has batch dimension, using first item.")
            samples_tensor = samples_tensor[0]
        elif samples_tensor.dim() != 2: # (channels, samples)
             raise ValueError(f"Unexpected audio tensor dimensions: {samples_tensor.shape}. Expected (channels, samples) or (batch, channels, samples).")

        # Transpose from (channels, samples) to (samples, channels) for scipy
        samples_np = samples_tensor.numpy().T # Transpose here

        # Convert float tensor (typically -1.0 to 1.0) to int16 for standard WAV
        # Check range before scaling to avoid clipping valid int16 audio
        if np.issubdtype(samples_np.dtype, np.floating):
             print(f"{log_prefix()} Converting float audio to int16 for WAV export.")
             samples_np = np.clip(samples_np, -1.0, 1.0) # Ensure range
             samples_np = (samples_np * 32767).astype(np.int16)
        elif not np.issubdtype(samples_np.dtype, np.integer):
             print(f"WARN: {log_prefix()} Audio tensor is not float or integer type ({samples_np.dtype}). WAV export might be incorrect.")
             # Attempt conversion anyway, but it might fail or be wrong
             samples_np = samples_np.astype(np.int16)


        # Generate temporary file path
        output_dir = folder_paths.get_temp_directory()
        os.makedirs(output_dir, exist_ok=True)
        filename = f"fal_omni_temp_audio_{uuid.uuid4().hex}.wav"
        temp_audio_filepath = os.path.join(output_dir, filename)

        # Write the WAV file
        print(f"{log_prefix()} Saving temporary WAV file: {temp_audio_filepath}")
        scipy.io.wavfile.write(temp_audio_filepath, sample_rate, samples_np)

        return temp_audio_filepath

    except Exception as e:
        print(f"ERROR: {log_prefix()} Failed to save audio tensor to temporary WAV: {e}")
        traceback.print_exc()
        return None

    

class FalAPIOmniProNode:
    # Define the standard keys this node will use when injecting media URLs
    # Users should generally avoid using these keys in their `parameters_json`
    # unless they specifically want to override the auto-uploaded file.
    AUTO_KEY_START_IMAGE = "image_url"
    AUTO_KEY_END_IMAGE = "end_image_url" # Use a distinct key
    AUTO_KEY_INPUT_VIDEO = "video_url"
    AUTO_KEY_INPUT_AUDIO = "audio_url"

    @classmethod # <-- Only one decorator now
    def INPUT_TYPES(cls):
        return {
            "required": {
                # --- ADD THESE LINES BACK ---
                "endpoint_id": ("STRING", {
                    "multiline": False,
                    "default": "fal-ai/some-model/endpoint-id"
                }),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "Paste FAL_KEY credentials here (e.g., key_id:key_secret)"
                }),
                # --- END OF ADDED LINES ---
                "parameters_json": ("STRING", {
                    "multiline": True,
                    "default": json.dumps({
                        "prompt": "A description for the model",
                        "seed": 12345,
                        "num_inference_steps": 25,
                    }, indent=2)
                }),
            },
            "optional": {
                "start_image": ("IMAGE",),
                "end_image": ("IMAGE",),
                "input_video": ("IMAGE",),
                "input_audio": ("AUDIO",),
                "cleanup_temp_files": ("BOOLEAN", {"default": True}),
                "output_video_fps": ("INT", {"default": 30, "min": 1, "max": 120}),
            }
        }
        
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_batch",)
    FUNCTION = "execute_omni_request"
    CATEGORY = "BS_FalAi-API-Omni" # Or your existing category

    # --- Main Execution Method ---
    def execute_omni_request(self, endpoint_id, api_key, parameters_json,
                             start_image=None, end_image=None, input_video=None,
                             input_audio=None, # Changed from audio_file_path
                             cleanup_temp_files=True,
                             output_video_fps=30):

        print(f"{log_prefix()} Starting Omni Pro request execution.")
        uploaded_media_urls = {} # Stores URLs keyed by AUTO_KEY_* constants
        temp_files_to_clean = []
        final_payload = {}

        # --- 1. API Key Setup ---
        if not api_key or not api_key.strip() or api_key == "Paste FAL_KEY credentials here (e.g., key_id:key_secret)":
            print(f"ERROR: {log_prefix()} API Key input field is empty or contains default text.")
            return (None,)
        api_key_value = api_key.strip()
        # Basic format check (optional but helpful)
        if ':' not in api_key_value and len(api_key_value) < 20:
             print(f"WARN: {log_prefix()} API Key doesn't seem to follow the usual 'key_id:key_secret' format.")
        try:
            os.environ["FAL_KEY"] = api_key_value
            print(f"{log_prefix()} Using provided API Key.")
        except Exception as e:
             print(f"ERROR: {log_prefix()} Failed to set API Key environment variable: {e}")
             traceback.print_exc(); return (None,)

        # --- 2. Parse User Parameters JSON ---
        user_params = {}
        try:
            if parameters_json and parameters_json.strip():
                user_params = json.loads(parameters_json)
                if not isinstance(user_params, dict):
                    raise ValueError("Parameters JSON did not parse into a dictionary.")
                print(f"{log_prefix()} Successfully parsed parameters JSON.")
            else:
                print(f"{log_prefix()} No parameters provided in JSON input, proceeding.")
        except json.JSONDecodeError as e:
            print(f"ERROR: {log_prefix()} Invalid JSON in 'parameters_json' input: {e}")
            print(f"--- Received JSON string: ---\n{parameters_json}\n-----------------------------")
            return (None,)
        except ValueError as e:
            print(f"ERROR: {log_prefix()} Error processing parameters JSON: {e}")
            return (None,)

        # --- 3. Handle and Upload Media Inputs (Automatic) ---
        upload_error = False
        try:
            # --- Start Image ---
            if start_image is not None:
                print(f"{log_prefix()} Processing start_image input...")
                img_bytes, content_type = _prepare_image_bytes(start_image)
                if img_bytes and content_type:
                    url = _upload_media_to_fal(img_bytes, "start_image.png", content_type)
                    if url:
                        uploaded_media_urls[self.AUTO_KEY_START_IMAGE] = url
                    else: upload_error = True; print(f"ERROR: {log_prefix()} Failed to upload start_image.")
                else: upload_error = True; print(f"ERROR: {log_prefix()} Failed to prepare start_image bytes.")

            # --- End Image ---
            if end_image is not None and not upload_error: # Skip if previous uploads failed
                print(f"{log_prefix()} Processing end_image input...")
                img_bytes, content_type = _prepare_image_bytes(end_image)
                if img_bytes and content_type:
                    url = _upload_media_to_fal(img_bytes, "end_image.png", content_type)
                    if url:
                        uploaded_media_urls[self.AUTO_KEY_END_IMAGE] = url
                    else: upload_error = True; print(f"ERROR: {log_prefix()} Failed to upload end_image.")
                else: upload_error = True; print(f"ERROR: {log_prefix()} Failed to prepare end_image bytes.")

            # --- Input Video ---
            if input_video is not None and not upload_error:
                print(f"{log_prefix()} Processing input_video (Image Batch)...")
                temp_video_path = _save_tensor_to_temp_video(input_video, fps=output_video_fps) # Use FPS input
                if temp_video_path and os.path.exists(temp_video_path):
                    temp_files_to_clean.append(temp_video_path) # Mark for cleanup AFTER potential use
                    try:
                        with open(temp_video_path, 'rb') as vf:
                            video_bytes = vf.read()
                        if video_bytes:
                            content_type = "video/mp4" # We saved as MP4
                            url = _upload_media_to_fal(video_bytes, os.path.basename(temp_video_path), content_type)
                            if url:
                                uploaded_media_urls[self.AUTO_KEY_INPUT_VIDEO] = url
                            else: upload_error = True; print(f"ERROR: {log_prefix()} Failed to upload input_video.")
                        else: upload_error = True; print(f"ERROR: {log_prefix()} Failed to read bytes from temporary video file.")
                    except Exception as read_e:
                        upload_error = True; print(f"ERROR: {log_prefix()} Failed to read temp video file {temp_video_path}: {read_e}")
                else: upload_error = True; print(f"ERROR: {log_prefix()} Failed to save input_video tensor to temporary file.")

            # --- Input Audio ---
            if input_audio is not None and not upload_error:
                print(f"{log_prefix()} Processing input_audio (AUDIO Tensor)...")
                # Save the audio tensor to a temporary WAV file first
                temp_audio_path = _save_audio_tensor_to_temp_wav(input_audio)

                if temp_audio_path and os.path.exists(temp_audio_path):
                    # Mark the temp WAV for cleanup later
                    temp_files_to_clean.append(temp_audio_path)
                    try:
                        # Read the bytes from the temp WAV file we just created
                        with open(temp_audio_path, 'rb') as af:
                            audio_bytes = af.read()

                        if audio_bytes:
                            # We know it's WAV, pass appropriate content type
                            content_type = "audio/wav"
                            # Pass the bytes and filename hint to the upload helper
                            # _upload_media_to_fal will handle creating *another* temp file
                            # specifically for the fal_client.upload_file call.
                            url = _upload_media_to_fal(audio_bytes, os.path.basename(temp_audio_path), content_type)
                            if url:
                                uploaded_media_urls[self.AUTO_KEY_INPUT_AUDIO] = url
                            else:
                                upload_error = True
                                print(f"ERROR: {log_prefix()} Failed to upload input_audio (from tensor).")
                        else:
                            upload_error = True
                            print(f"ERROR: {log_prefix()} Failed to read bytes from temporary audio file: {temp_audio_path}")
                    except Exception as read_e:
                         upload_error = True
                         print(f"ERROR: {log_prefix()} Failed to read temp audio file {temp_audio_path}: {read_e}")
                         traceback.print_exc()
                else:
                    # Failed to save the tensor to a temp file
                    upload_error = True
                    print(f"ERROR: {log_prefix()} Failed to save input_audio tensor to temporary WAV file.")

        except Exception as e:
             # Catch unexpected errors during the media processing phase
             print(f"ERROR: {log_prefix()} Unexpected error during media preparation or upload: {e}")
             traceback.print_exc()
             upload_error = True # Ensure we skip API call

        # If any upload failed, stop before making the API call
        if upload_error:
             print(f"ERROR: {log_prefix()} Aborting due to media processing/upload errors.")
             # Perform early cleanup
             if cleanup_temp_files:
                 for temp_file in temp_files_to_clean:
                     if os.path.exists(temp_file):
                         try: os.remove(temp_file)
                         except Exception as clean_e: print(f"WARN: {log_prefix()} Early cleanup failed: {clean_e}")
             return (None,)

        # --- 4. Construct Final Payload (Merge User Params + Auto Media URLs) ---
        final_payload = user_params.copy() # Start with user's non-media params

        print(f"{log_prefix()} Injecting uploaded media URLs into payload...")
        for auto_key, url in uploaded_media_urls.items():
            if auto_key in final_payload:
                print(f"WARN: {log_prefix()} User-provided key '{auto_key}' in JSON conflicts with auto-injected media URL. Overwriting with uploaded URL: {url}")
            else:
                print(f"{log_prefix()} Injecting '{auto_key}': '{url}'")
            final_payload[auto_key] = url

        # print(f"{log_prefix()} Final Payload: {json.dumps(final_payload, indent=2)}") # Debug: View the final payload

        # --- 5. API Call, Download, Frame Extraction ---
        request_id = None
        result_url = None
        result_content_type = None
        temp_download_filepath = None

        try:
            print(f"{log_prefix()} Submitting job to endpoint: {endpoint_id}")
            if not endpoint_id or not endpoint_id.strip():
                 raise ValueError("Endpoint ID cannot be empty.")

            # Submit the job
            handler = fal_client.submit(endpoint_id.strip(), arguments=final_payload)
            request_id = handler.request_id
            print(f"{log_prefix()} Job submitted. Request ID: {request_id}")
            print(f"{log_prefix()} Waiting for Fal.ai job completion... (This may take time)")

            # Poll for the result (blocking)
            response = handler.get() # This blocks until completion
            print(f"{log_prefix()} Fal.ai job completed.")
            # print(f"{log_prefix()} Full response: {json.dumps(response, indent=2)}") # Debug raw response

            # --- Process Response ---
            # Flexible response parsing (same logic as before)
            if isinstance(response, dict):
                # Check common video structures
                if 'video' in response and isinstance(response['video'], dict) and 'url' in response['video']:
                    result_url = response['video']['url']
                    result_content_type = response['video'].get('content_type', 'video/mp4')
                elif 'videos' in response and isinstance(response['videos'], list) and len(response['videos']) > 0:
                     item = response['videos'][0]
                     if isinstance(item, dict) and 'url' in item:
                         result_url = item['url']
                         result_content_type = item.get('content_type', 'video/mp4')
                # Check common image structures
                elif 'image' in response and isinstance(response['image'], dict) and 'url' in response['image']:
                    result_url = response['image']['url']
                    result_content_type = response['image'].get('content_type', 'image/png')
                elif 'images' in response and isinstance(response['images'], list) and len(response['images']) > 0:
                     item = response['images'][0]
                     if isinstance(item, dict) and 'url' in item:
                         result_url = item['url']
                         result_content_type = item.get('content_type', 'image/png')
                # Check for a direct URL at the top level
                elif 'url' in response and isinstance(response['url'], str):
                    result_url = response['url']
                    result_content_type = response.get('content_type') # Might be None
                    # Guess content type if missing
                    if not result_content_type:
                        guessed_type, _ = mimetypes.guess_type(result_url)
                        if guessed_type:
                            result_content_type = guessed_type
                        else: # Basic extension check if mimetypes fails
                             if any(ext in result_url.lower() for ext in ['.mp4', '.webm', '.mov', '.avi']): result_content_type = 'video/mp4'
                             elif any(ext in result_url.lower() for ext in ['.png', '.jpg', '.jpeg', '.webp']): result_content_type = 'image/png'

            if not result_url:
                 print(f"WARN: {log_prefix()} Could not find a usable media 'url' in Fal.ai response.")
                 print(f"--- Full response received: --- \n{json.dumps(response, indent=2)}\n-----------------------------")
                 print(f"{log_prefix()} Returning empty result as no output media URL was found.")
                 return (None,) # Indicate no image/video output

            print(f"{log_prefix()} Result URL found: {result_url}")
            print(f"{log_prefix()} Result Content-Type: {result_content_type or 'Unknown'}")

            # --- Download the Result ---
            print(f"{log_prefix()} Downloading result file...")
            media_response = requests.get(result_url, stream=True, timeout=600) # Long timeout
            media_response.raise_for_status() # Check for HTTP errors

            # Determine if it's video or image based on content type hint
            # Prioritize official content-type if available
            is_video = False
            is_image = False
            if result_content_type:
                if 'video' in result_content_type.lower(): is_video = True
                elif 'image' in result_content_type.lower(): is_image = True

            # Fallback to URL extension check if content type was inconclusive
            if not is_video and not is_image:
                 print(f"WARN: {log_prefix()} Content type ambiguous. Guessing based on URL extension.")
                 if any(ext in result_url.lower() for ext in ['.mp4', '.webm', '.mov', '.avi']):
                      is_video = True
                      print(f"{log_prefix()} Guessed type: VIDEO based on extension.")
                 elif any(ext in result_url.lower() for ext in ['.png', '.jpg', '.jpeg', '.webp']):
                      is_image = True
                      print(f"{log_prefix()} Guessed type: IMAGE based on extension.")

            # --- Process Downloaded Media ---
            if is_video:
                print(f"{log_prefix()} Result identified as VIDEO. Processing...")
                output_dir = folder_paths.get_temp_directory()
                os.makedirs(output_dir, exist_ok=True)
                extension = '.mp4' # Default
                if result_content_type and 'video/webm' in result_content_type: extension = '.webm'
                elif result_url.lower().endswith('.webm'): extension = '.webm'
                # Add more specific extensions if needed

                filename = f"fal_omni_result_{uuid.uuid4().hex}{extension}"
                temp_download_filepath = os.path.join(output_dir, filename)
                temp_files_to_clean.append(temp_download_filepath) # Mark for cleanup

                with open(temp_download_filepath, 'wb') as f_out:
                    for chunk in media_response.iter_content(chunk_size=1024*1024): f_out.write(chunk)
                print(f"{log_prefix()} Video downloaded to: {temp_download_filepath}")

                # Extract Frames using OpenCV
                print(f"{log_prefix()} Extracting frames...")
                frames_list = []
                cap = cv2.VideoCapture(temp_download_filepath)
                if not cap.isOpened(): raise IOError(f"Could not open downloaded video file: {temp_download_filepath}")
                frame_count = 0
                while True:
                    ret, frame = cap.read()
                    if not ret: break
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames_list.append(frame_rgb)
                    frame_count += 1
                cap.release()
                if not frames_list: raise ValueError(f"No frames extracted from video: {temp_download_filepath}")
                print(f"{log_prefix()} Extracted {frame_count} frames.")

                # Convert to Tensor Batch
                frames_np = np.stack(frames_list, axis=0) # (B, H, W, C)
                frames_tensor = torch.from_numpy(frames_np).float() / 255.0 # Normalize
                print(f"{log_prefix()} Frames tensor shape: {frames_tensor.shape}")
                return (frames_tensor,)

            elif is_image:
                print(f"{log_prefix()} Result identified as IMAGE. Processing...")
                image_bytes = media_response.content
                pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                img_np = np.array(pil_image, dtype=np.float32) / 255.0 # H, W, C float 0-1
                img_tensor = torch.from_numpy(img_np).unsqueeze(0) # Add batch dim -> B, H, W, C
                print(f"{log_prefix()} Image tensor shape: {img_tensor.shape}")
                return (img_tensor,)

            else:
                 # If we couldn't determine the type after checks
                 print(f"ERROR: {log_prefix()} Could not determine if the result URL is a video or an image.")
                 print(f"--- URL: {result_url}")
                 print(f"--- Content-Type: {result_content_type or 'Unknown'}")
                 # Try downloading as generic file? For now, fail.
                 return(None,)

        # --- Exception Handling ---
        except requests.exceptions.RequestException as e:
             url_for_error = result_url if result_url else f"API Endpoint: {endpoint_id}"
             print(f"ERROR: {log_prefix()} Network request failed ({url_for_error}): {e}")
             traceback.print_exc(); return (None,)
        except (cv2.error, IOError, ValueError, Image.UnidentifiedImageError) as e:
             print(f"ERROR: {log_prefix()} Media processing/conversion error: {e}")
             traceback.print_exc(); return (None,)
        except Exception as e:
            req_id_str = f"Request ID: {request_id}" if request_id else "Request ID: N/A (submission failed?)"
            print(f"ERROR: {log_prefix()} Unexpected error during API call or processing ({req_id_str}): {e}")
            if endpoint_id and ("not found" in str(e).lower() or "404" in str(e)):
                 print(f"--- Hint: Check if the Endpoint ID '{endpoint_id}' is correct and the API key is valid. ---")
            elif "Request entity too large" in str(e):
                 print(f"--- Hint: Uploaded files might exceed size limits for '{endpoint_id}'. ---")
            traceback.print_exc(); return (None,)

        # --- Final Cleanup ---
        finally:
            if cleanup_temp_files:
                print(f"{log_prefix()} Cleaning up temporary files...")
                for temp_file in temp_files_to_clean:
                    if temp_file and os.path.exists(temp_file):
                        try:
                            print(f"{log_prefix()} Removing: {temp_file}")
                            os.remove(temp_file)
                        except Exception as e:
                            print(f"WARN: {log_prefix()} Failed to delete temp file '{temp_file}': {e}")
            else:
                 if temp_files_to_clean:
                      print(f"{log_prefix()} Skipping cleanup for temporary files:")
                      for temp_file in temp_files_to_clean:
                           if temp_file and os.path.exists(temp_file): print(f" - {temp_file}")

            # Optional: Unset the environment variable if needed, though fal_client might cache it.
            # if "FAL_KEY" in os.environ:
            #     try: del os.environ["FAL_KEY"]
            #     except Exception: pass
            #     print(f"{log_prefix()} Unset FAL_KEY environment variable.")


NODE_CLASS_MAPPINGS = {
    "FalAPIVideoGeneratorI2V": FalAPIVideoGeneratorI2V,
    "FalAPIVideoGeneratorT2V": FalAPIVideoGeneratorT2V,
    "FalAPIOmniProNode": FalAPIOmniProNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FalAPIVideoGeneratorI2V": "FAL AI Image-to-Video",
    "FalAPIVideoGeneratorT2V": "FAL AI Text-to-Video",
    "FalAPIOmniProNode": "fal.ai API Omni Pro Node"
}
