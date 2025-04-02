import os
import io
import base64
import uuid
import traceback
import requests
import numpy as np
import torch
import fal_client # Assuming fal_client is installed: pip install fal
from PIL import Image
import cv2 # Requires opencv-python: pip install opencv-python

# Helper to access ComfyUI's path functions
import folder_paths

# --- Configuration Data from the Table (Keep as is) ---
MODEL_CONFIGS = {
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
        # Note: end_image_url is not handled in this basic version
        # Note: loop is in schema but removed from INPUTS per request
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
         # Note: 'motion' not added as a standard input, 'num_inference_steps' not added (but 'steps' is)
         # Note: frames_per_second is in schema but removed from INPUTS per request
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
        # Note: num_inference_steps is in schema, but 'steps' is the input name
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
    # --- Add TEXT-TO-VIDEO models here if needed, adjusting schemas ---
    # Example based on original Kling node (needs verification)
     "Kling 1.6 Text to Video (Pro)": {
        "endpoint": "fal-ai/kling-video/v1.6/pro/text-to-video", # Hypothetical endpoint
        "resolutions": [],
        "aspect_ratios": ["16:9", "9:16", "1:1"], # Guessing
        "durations": [5, 10], # Guessing
        "schema_str": "[Prompt:String], [negative_prompt:String], [duration:DurationEnum], [aspect_ratio:AspectRatioEnum], [seed:Integer], [width:Integer], [height:Integer], [motion_bucket_id:Integer], [cond_aug:Float], [steps:Integer], [guidance_scale:Float], [fps:Integer]", # More comprehensive guess
        # Note: motion_bucket_id, cond_aug, fps are in schema but removed from INPUTS per request
     }
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


# --- Populate Model Configs with Parsed Schema (Keep as is) ---
for name, config in MODEL_CONFIGS.items():
    config['expected_params'] = parse_schema(config['schema_str'])

# --- Dynamically Create Dropdown Lists (Keep as is) ---
ALL_MODEL_NAMES = sorted(list(MODEL_CONFIGS.keys()))
ALL_RESOLUTIONS = sorted(list(set(res for cfg in MODEL_CONFIGS.values() for res in cfg['resolutions'])))
ALL_ASPECT_RATIOS = sorted(list(set(ar for cfg in MODEL_CONFIGS.values() for ar in cfg['aspect_ratios'])))
# Add common defaults if lists are empty
if not ALL_RESOLUTIONS: ALL_RESOLUTIONS = ["720p", "1080p", "512p", "576p"] # Add common ones
if not ALL_ASPECT_RATIOS: ALL_ASPECT_RATIOS = ["16:9", "9:16", "1:1", "4:3", "3:4"]

# Add a "None" or "Auto" option where appropriate
if "auto" not in ALL_ASPECT_RATIOS: ALL_ASPECT_RATIOS.insert(0, "auto")
ALL_RESOLUTIONS.insert(0, "auto") # Use 'auto' as default/unspecified

# --- Define the Node Class ---
class FalAPIVideoGenerator:
    # --- INPUT_TYPES MODIFIED ---
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (ALL_MODEL_NAMES,),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "Paste FAL_KEY credentials here (e.g., key_id:key_secret)"
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "prompt": ("STRING", {"multiline": True, "default": "A wild Burgstall appears"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "negative_prompt": ("STRING", {"multiline": True, "default": "Ugly, blurred, distorted"}),
                "resolution_enum": (ALL_RESOLUTIONS, {"default": "auto"}),
                "aspect_ratio_enum": (ALL_ASPECT_RATIOS, {"default": "auto"}),
                # width and height are correctly removed
                "duration_seconds": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 30.0, "step": 0.5}),
                "guidance_scale": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 100, "step": 1}),
                # fps REMOVED
                # motion_bucket_id REMOVED
                # cond_aug REMOVED
                "num_frames": ("INT", {"default": 0, "min": 0, "max": 1000}), # Keep if any model uses it (e.g., Wan)
                # loop REMOVED
                "style": (["auto", "cinematic", "anime", "photorealistic", "fantasy", "cartoon"], {"default": "auto"}), # Keep if any model uses it (e.g., PixVerse)
                "prompt_optimizer": ("BOOLEAN", {"default": False}), # Keep if any model uses it (e.g., MiniMax Director/Live)
                "cleanup_temp_video": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_batch",)
    FUNCTION = "generate_video"
    CATEGORY = "BS_FalAi-API-Video" # Or your preferred category

    # --- _prepare_image method (Keep as is, it's good) ---
    def _prepare_image(self, image_tensor, target_width=None, target_height=None):
        """Converts ComfyUI Image Tensor to Base64 Data URI, resizing if needed."""
        if image_tensor is None:
            print("FalAPIVideoGenerator: No image provided to _prepare_image.")
            return None

        print("FalAPIVideoGenerator: Preparing image...")
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
                print(f"FalAPIVideoGenerator: Resizing input image from {pil_image.width}x{pil_image.height} to {target_width}x{target_height}")
                pil_image = pil_image.resize((target_width, target_height), Image.LANCZOS)

            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG")
            img_bytes = buffered.getvalue()
            img_b64_encoded = base64.b64encode(img_bytes).decode('utf-8')
            img_data_uri = f"data:image/png;base64,{img_b64_encoded}"
            print("FalAPIVideoGenerator: Image preparation complete.")
            return img_data_uri

        except Exception as e:
            print(f"ERROR: FalAPIVideoGenerator: Image processing failed: {e}")
            traceback.print_exc()
            return None

    # --- generate_video method MODIFIED ---
    def generate_video(self, model_name, api_key, seed, prompt,
                         image=None, negative_prompt=None,
                         resolution_enum="auto", aspect_ratio_enum="auto",
                         # width and height parameters correctly removed
                         duration_seconds=5.0,
                         guidance_scale=7.5, steps=25, # fps removed
                         # motion_bucket_id removed
                         # cond_aug removed
                         num_frames=0, # loop removed
                         style="auto",
                         prompt_optimizer=False, cleanup_temp_video=True):

        # --- 1. Get Model Config ---
        if model_name not in MODEL_CONFIGS:
            print(f"ERROR: FalAPIVideoGenerator: Unknown model name '{model_name}'.")
            return (None,)
        config = MODEL_CONFIGS[model_name]
        endpoint_id = config['endpoint']
        expected_params = config['expected_params']
        print(f"FalAPIVideoGenerator: Selected Model: {model_name}")
        print(f"FalAPIVideoGenerator: Endpoint: {endpoint_id}")
        # print(f"FalAPIVideoGenerator: Expected params by model: {expected_params}") # Optional debug

        # --- 2. API Key Setup ---
        if not api_key or not api_key.strip() or api_key == "Paste FAL_KEY credentials here (e.g., key_id:key_secret)":
            print("ERROR: FalAPIVideoGenerator: API Key input field is empty or contains default text. Please paste your actual FAL key.")
            return (None,)
        api_key_value = api_key.strip()
        if ':' not in api_key_value and len(api_key_value) < 20:
             print("WARN: FalAPIVideoGenerator: API Key doesn't seem to follow the usual 'key_id:key_secret' format. Ensure it's correct.")
        try:
            # Set environment variable ONLY for the duration of this call?
            # Consider implications if running multiple FAL nodes concurrently.
            # For simplicity here, we set it globally for the process.
            os.environ["FAL_KEY"] = api_key_value
            print("FalAPIVideoGenerator: Using provided API Key.")
        except Exception as e:
             print(f"ERROR: FalAPIVideoGenerator: Failed to set API Key environment variable: {e}")
             traceback.print_exc(); return (None,)

        # --- 3. Prepare Payload Dynamically ---
        payload = {}
        img_data_uri = None

        # --- Image Handling ---
        if 'image_url' in expected_params:
            if image is not None:
                # Call _prepare_image without target width/height
                img_data_uri = self._prepare_image(image, target_width=None, target_height=None)
                if img_data_uri:
                    payload['image_url'] = img_data_uri
                else:
                    print("ERROR: FalAPIVideoGenerator: Failed to prepare image, aborting.")
                    return (None,)
            else:
                # Image is expected but not provided - this might be an error depending on the specific endpoint
                print(f"WARN: FalAPIVideoGenerator: Model '{model_name}' expects 'image_url', but no image was provided.")
                # Decide if this should be a hard error: return (None,)
        elif image is not None:
             print(f"WARN: FalAPIVideoGenerator: Image provided, but model '{model_name}' does not seem to expect 'image_url'. Image ignored.")

        # --- Text Prompts ---
        if 'prompt' in expected_params:
            if prompt and prompt.strip():
                 payload['prompt'] = prompt.strip()
            else:
                 # Prompt is usually required if expected
                 print(f"WARN: FalAPIVideoGenerator: Model '{model_name}' expects 'prompt', but it's empty.")
                 # Decide if this should be a hard error: return (None,)
        if 'negative_prompt' in expected_params and negative_prompt and negative_prompt.strip():
            payload['negative_prompt'] = negative_prompt.strip()

        # --- Numeric/Boolean/Enum Parameters (Add ONLY if expected) ---
        # Note: The key in 'payload' should match the API expectation (e.g., 'duration', not 'duration_seconds')
        # We need to map input names back to potential API names if they differ.
        param_map = {
            "seed": seed,
            "duration_seconds": duration_seconds, # Will map to 'duration' if needed below
            "guidance_scale": guidance_scale,
            "steps": steps, # Maps to 'steps' or 'num_inference_steps'
            "num_frames": num_frames,
            "style": style,
            "prompt_optimizer": prompt_optimizer,
            "resolution_enum": resolution_enum, # Maps to 'resolution'
            "aspect_ratio_enum": aspect_ratio_enum # Maps to 'aspect_ratio'
            # Add other parameters here if needed
        }

        api_name_map = {
            "duration_seconds": "duration",
            "steps": "num_inference_steps", # Map to this if 'steps' isn't the API name
            "resolution_enum": "resolution",
            "aspect_ratio_enum": "aspect_ratio"
        }

        for input_name, value in param_map.items():
            # Check if the direct input_name is expected
            if input_name in expected_params:
                 # Handle 'auto' or default values appropriately for enums/optional fields
                 if isinstance(value, str) and value.lower() == "auto":
                     # Decide whether to omit 'auto' or send it; depends on API. Omit for now.
                     print(f"FalAPIVideoGenerator: Skipping '{input_name}' parameter with value 'auto'.")
                     continue
                 if isinstance(value, int) and value == 0 and input_name in ["num_frames"]: # Example: Skip default 0 num_frames
                     print(f"FalAPIVideoGenerator: Skipping '{input_name}' parameter with default value {value}.")
                     continue
                 payload[input_name] = value
            else:
                 # Check if there's a mapped API name for this input
                 api_name = api_name_map.get(input_name)
                 if api_name and api_name in expected_params:
                     if isinstance(value, str) and value.lower() == "auto":
                         print(f"FalAPIVideoGenerator: Skipping '{api_name}' parameter (from '{input_name}') with value 'auto'.")
                         continue
                     if isinstance(value, int) and value == 0 and input_name in ["num_frames"]:
                         print(f"FalAPIVideoGenerator: Skipping '{api_name}' parameter (from '{input_name}') with default value {value}.")
                         continue

                     # Special case for duration which might be INT or FLOAT depending on API
                     if input_name == "duration_seconds" and "duration:Integer" in config['schema_str']:
                          payload[api_name] = int(value) # Convert to INT if API expects Integer
                     elif input_name == "steps" and "num_inference_steps:Integer" in config['schema_str']:
                         payload[api_name] = value # API expects 'num_inference_steps'
                     else:
                         payload[api_name] = value # Use original value

        # --- Explicitly handle width/height if model expects them (using enums) ---
        # This logic remains complex as width/height inputs are removed.
        # We rely purely on resolution/aspect ratio enums if the model needs dimensions.
        used_enum_res = False
        if 'resolution_enum' in expected_params and resolution_enum != "auto":
            payload['resolution'] = resolution_enum # Assumes API expects the enum string
            used_enum_res = True
            # Remove width/height if they were somehow added, as enum takes precedence
            payload.pop('width', None)
            payload.pop('height', None)
        elif 'resolution' in expected_params and resolution_enum != "auto":
             payload['resolution'] = resolution_enum
             used_enum_res = True
             payload.pop('width', None)
             payload.pop('height', None)


        used_enum_ar = False
        if 'aspect_ratio_enum' in expected_params and aspect_ratio_enum != "auto":
            payload['aspect_ratio'] = aspect_ratio_enum # Assumes API expects the enum string
            used_enum_ar = True
            # Remove width/height if they were somehow added, as enum takes precedence
            payload.pop('width', None)
            payload.pop('height', None)
        elif 'aspect_ratio' in expected_params and aspect_ratio_enum != "auto":
             payload['aspect_ratio'] = aspect_ratio_enum
             used_enum_ar = True
             payload.pop('width', None)
             payload.pop('height', None)

        # Warn if width/height are expected but no enums provided resolution
        # This might indicate the user needs to use a model that doesn't require explicit dims
        # or that our enum handling isn't sufficient for this specific model.
        if 'width' in expected_params and not used_enum_res and not used_enum_ar:
            print(f"WARN: FalAPIVideoGenerator: Model '{model_name}' expects 'width', but no explicit width/height inputs exist and resolution/aspect ratio enums are 'auto' or not used. API might use defaults or fail.")
        if 'height' in expected_params and not used_enum_res and not used_enum_ar:
            print(f"WARN: FalAPIVideoGenerator: Model '{model_name}' expects 'height', but no explicit width/height inputs exist and resolution/aspect ratio enums are 'auto' or not used. API might use defaults or fail.")


        # --- 4. API Call, Download, Frame Extraction (Using FalKlingVideoToFrames Logic) ---
        request_id = None
        video_url = None
        temp_video_filepath = None # Define filepath here for broader scope

        try:
            print(f"FalAPIVideoGenerator: Submitting job to endpoint: {endpoint_id}")
            # print(f"FalAPIVideoGenerator: Payload: {payload}") # Debug: Print final payload
            if not endpoint_id or not endpoint_id.strip():
                 raise ValueError("Endpoint ID cannot be empty.")

            # --- API Call using fal_client.submit ---
            handler = fal_client.submit(endpoint_id.strip(), arguments=payload)
            request_id = handler.request_id
            print(f"FalAPIVideoGenerator: Job submitted. Request ID: {request_id}")
            print("FalAPIVideoGenerator: Waiting for Fal.ai job completion...")
            response = handler.get() # Blocks until completion
            print("FalAPIVideoGenerator: Fal.ai job completed.")

            # --- Process Response to find Video URL ---
            # Try common patterns for video URL in response
            video_data = response.get('video') # Common pattern
            if isinstance(video_data, dict):
                video_url = video_data.get('url')
            elif isinstance(response.get('videos'), list) and len(response['videos']) > 0: # Another pattern
                 vid_info = response['videos'][0]
                 if isinstance(vid_info, dict):
                     video_url = vid_info.get('url')
            elif isinstance(response, dict) and response.get('url') and isinstance(response.get('url'), str): # Top-level URL?
                 # Check if it looks like a video URL
                 if any(ext in response['url'].lower() for ext in ['.mp4', '.webm', '.mov', '.avi']):
                      video_url = response['url']

            if not video_url:
                 print(f"ERROR: FalAPIVideoGenerator: Could not find video 'url' in Fal.ai response structure.")
                 print(f"Full response: {response}")
                 return (None,)

            print(f"FalAPIVideoGenerator: Video URL received: {video_url}")
            print("FalAPIVideoGenerator: Downloading video file...")

            # --- Download Video ---
            video_response = requests.get(video_url, stream=True, timeout=300) # Increased timeout
            video_response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            # --- Save to Temporary File ---
            output_dir = folder_paths.get_temp_directory()
            os.makedirs(output_dir, exist_ok=True)
            # Determine extension more robustly
            content_type = video_response.headers.get('content-type', '').lower()
            extension = '.mp4' # Default
            if 'video/mp4' in content_type: extension = '.mp4'
            elif 'video/webm' in content_type: extension = '.webm'
            elif video_url.lower().endswith('.mp4'): extension = '.mp4'
            elif video_url.lower().endswith('.webm'): extension = '.webm'

            filename = f"fal_api_video_temp_{uuid.uuid4().hex}{extension}"
            temp_video_filepath = os.path.join(output_dir, filename)

            with open(temp_video_filepath, 'wb') as video_file:
                for chunk in video_response.iter_content(chunk_size=1024*1024): # Download in chunks
                    video_file.write(chunk)
            print(f"FalAPIVideoGenerator: Video downloaded to temporary file: {temp_video_filepath}")

            # --- Extract Frames using OpenCV ---
            print(f"FalAPIVideoGenerator: Extracting frames from {temp_video_filepath}...")
            frames_list = []
            cap = cv2.VideoCapture(temp_video_filepath)
            if not cap.isOpened():
                # Try to clean up before failing
                if cleanup_temp_video and temp_video_filepath and os.path.exists(temp_video_filepath):
                    try: os.remove(temp_video_filepath)
                    except Exception as clean_e: print(f"WARN: FalAPIVideoGenerator: Failed to cleanup partially downloaded/corrupt file: {clean_e}")
                raise IOError(f"Could not open video file: {temp_video_filepath}")

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break # End of video or error

                # Convert frame from BGR (OpenCV default) to RGB (ComfyUI expects)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_list.append(frame_rgb)
                frame_count += 1

            cap.release() # Release the video capture object

            if not frames_list:
                 # Clean up the (likely empty or corrupt) video file
                if cleanup_temp_video and temp_video_filepath and os.path.exists(temp_video_filepath):
                    try: os.remove(temp_video_filepath)
                    except Exception as clean_e: print(f"WARN: FalAPIVideoGenerator: Failed to cleanup empty video file: {clean_e}")
                raise ValueError(f"No frames extracted from video: {temp_video_filepath}")

            print(f"FalAPIVideoGenerator: Extracted {frame_count} frames.")

            # --- Convert Frames to Tensor Batch ---
            frames_np = np.stack(frames_list, axis=0)
            # Normalize pixel values from [0, 255] to [0.0, 1.0] and convert to float tensor
            frames_tensor = torch.from_numpy(frames_np).float() / 255.0
            print(f"FalAPIVideoGenerator: Frames converted to tensor with shape: {frames_tensor.shape}")

            # --- Return the Tensor Batch ---
            return (frames_tensor,)

        # --- Exception Handling (Adapted from FalKlingVideoToFrames) ---
        except requests.exceptions.RequestException as e:
             url_for_error = video_url if video_url else f"API Endpoint: {endpoint_id}"
             print(f"ERROR: FalAPIVideoGenerator: Network request failed ({url_for_error}): {e}")
             traceback.print_exc(); return (None,)
        except (cv2.error, IOError, ValueError) as e: # Catch OpenCV, File IO, or Value errors from extraction/conversion
             print(f"ERROR: FalAPIVideoGenerator: Video processing/Frame extraction error: {e}")
             traceback.print_exc(); return (None,)
        except Exception as e: # Catch-all for other errors (API errors, unexpected issues)
            req_id_str = request_id if request_id else 'N/A'
            print(f"ERROR: FalAPIVideoGenerator: Unexpected error (Request ID: {req_id_str}): {e}")
            # Add specific checks for Fal errors if the fal_client library raises unique exceptions
            # Example: if isinstance(e, fal_client.SomeFalError): ...
            if endpoint_id and ("not found" in str(e).lower() or "404" in str(e)):
                 print("--- Hint: Check if the Endpoint ID is correct and accessible. ---")
            # Print traceback for debugging any unexpected error
            traceback.print_exc(); return (None,) # Return None tuple on failure

        finally:
            # --- Cleanup Temporary Video File ---
            if cleanup_temp_video and temp_video_filepath and os.path.exists(temp_video_filepath):
                try:
                    print(f"FalAPIVideoGenerator: Cleaning up temporary video file: {temp_video_filepath}")
                    os.remove(temp_video_filepath)
                except Exception as e:
                    # Log warning but don't fail the whole node if cleanup fails
                    print(f"WARN: FalAPIVideoGenerator: Failed to delete temporary video file '{temp_video_filepath}': {e}")
            elif temp_video_filepath and os.path.exists(temp_video_filepath):
                 # Explicitly state if the file is kept
                 print(f"FalAPIVideoGenerator: Keeping temporary video file (cleanup_temp_video=False): {temp_video_filepath}")


# --- ComfyUI Node Mappings (Keep as is) ---
NODE_CLASS_MAPPINGS = { "FalAPIVideoGenerator": FalAPIVideoGenerator }
NODE_DISPLAY_NAME_MAPPINGS = { "FalAPIVideoGenerator": "FAL AI Video Generator" }