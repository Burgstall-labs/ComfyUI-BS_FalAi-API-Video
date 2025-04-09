import os
import io
import base64
import uuid
import json
import time # Ensure time is imported
import mimetypes
import traceback
import requests
import numpy as np
import torch
import scipy.io.wavfile
import fal_client # Assuming fal_client is installed: pip install fal-client
from PIL import Image
import cv2 # Requires opencv-python: pip install opencv-python

# Helper to access ComfyUI's path functions
import folder_paths

# --- Configuration Data with Categories ---
# (MODEL_CONFIGS dictionary remains the same as provided in your last snippet)
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

# --- Polling Helper Function ---
# (Keep _poll_fal_job function as provided in the previous response)
def _poll_fal_job(endpoint_id, request_id, polling_interval=3, timeout=600):
    """
    Polls a fal.ai job status until completion, failure, or timeout.
    Handles KeyboardInterrupt for cancellation.

    Args:
        endpoint_id (str): The fal.ai endpoint ID.
        request_id (str): The job request ID.
        polling_interval (int): Seconds to wait between status checks.
        timeout (int): Maximum seconds to wait for the job.

    Returns:
        dict: The final result dictionary if the job completes successfully.

    Raises:
        KeyboardInterrupt: If interrupted during polling.
        TimeoutError: If the job exceeds the timeout duration.
        RuntimeError: If the fal.ai job reports an error status.
        Exception: For other unexpected errors during polling.
    """
    start_time = time.time()
    print(f"[Fal Poller] Started polling job {request_id} for endpoint {endpoint_id}...")
    print(f"[Fal Poller] Timeout set to {timeout}s, Interval {polling_interval}s.")

    while True:
        # --- Timeout Check ---
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
            print(f"ERROR: [Fal Poller] Job {request_id} exceeded timeout of {timeout} seconds.")
            raise TimeoutError(f"Fal.ai job {request_id} timed out after {timeout}s")

        # --- Status Check ---
        try:
            # Use fal_client.status to get the current status
            status_response = fal_client.status(endpoint_id, request_id, logs=False) # Don't fetch logs repeatedly
            status = status_response.get('status')
            queue_pos = status_response.get('queue_position')

            print(f"[Fal Poller] Job {request_id}: Status={status}, Queue={queue_pos if queue_pos is not None else 'N/A'}, Elapsed={elapsed_time:.1f}s")

            if status == "COMPLETED":
                print(f"[Fal Poller] Job {request_id} completed.")
                # Fetch the final result separately after completion is confirmed
                final_result = fal_client.result(endpoint_id, request_id)
                return final_result # Success!

            elif status in ["ERROR", "FAILED", "CANCELLED"]: # Add other failure states if known
                error_message = f"Fal.ai job {request_id} failed with status: {status}"
                # Optionally try to fetch logs once on failure if status_response allows it
                # logs = status_response.get('logs') # Or fetch logs separately
                # if logs: error_message += f"\nLogs: {logs}"
                print(f"ERROR: [Fal Poller] {error_message}")
                raise RuntimeError(error_message)

            elif status in ["IN_QUEUE", "IN_PROGRESS"]:
                # Wait before the next poll
                try:
                    time.sleep(polling_interval)
                except KeyboardInterrupt:
                    print(f"\nWARN: [Fal Poller] KeyboardInterrupt caught during sleep for job {request_id}. Attempting cancellation...")
                    # Re-raise KeyboardInterrupt so the calling function can handle fal cancellation
                    raise KeyboardInterrupt

            else:
                # Unknown status - treat as potential issue and wait? Or error?
                print(f"WARN: [Fal Poller] Job {request_id} has unknown status: {status}. Continuing poll.")
                try:
                    time.sleep(polling_interval)
                except KeyboardInterrupt:
                     print(f"\nWARN: [Fal Poller] KeyboardInterrupt caught during sleep (unknown status) for job {request_id}. Attempting cancellation...")
                     raise KeyboardInterrupt

        except KeyboardInterrupt: # Catch if Ctrl+C happens during API call itself
            print(f"\nWARN: [Fal Poller] KeyboardInterrupt caught during API call for job {request_id}. Attempting cancellation...")
            raise KeyboardInterrupt
        except Exception as e:
            print(f"ERROR: [Fal Poller] Unexpected error polling job {request_id}: {e}")
            traceback.print_exc()
            # Re-raise the exception to be handled by the calling function
            raise e

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
for category, models in MODEL_CONFIGS.items():
    for name, config in models.items():
        config['expected_params'] = parse_schema(config['schema_str'])

# --- Dynamically Create Dropdown Lists (Keep as is) ---
ALL_MODEL_NAMES_I2V = sorted(list(MODEL_CONFIGS["image_to_video"].keys()))
ALL_MODEL_NAMES_T2V = sorted(list(MODEL_CONFIGS["text_to_video"].keys()))
ALL_RESOLUTIONS = sorted(list(set(res for category in MODEL_CONFIGS.values() for cfg in category.values() for res in cfg['resolutions'])))
ALL_ASPECT_RATIOS = sorted(list(set(ar for category in MODEL_CONFIGS.values() for cfg in category.values() for ar in cfg['aspect_ratios'])))
if not ALL_RESOLUTIONS: ALL_RESOLUTIONS = ["720p", "1080p", "512p", "576p"]
if not ALL_ASPECT_RATIOS: ALL_ASPECT_RATIOS = ["16:9", "9:16", "1:1", "4:3", "3:4"]
if "auto" not in ALL_ASPECT_RATIOS: ALL_ASPECT_RATIOS.insert(0, "auto")
ALL_RESOLUTIONS.insert(0, "auto")


# --- Helper Functions with Corrected Logging ---

def _prepare_image_bytes(image_tensor):
    """Converts ComfyUI Image Tensor to PNG bytes."""
    if image_tensor is None:
        print("[Fal Helper] No image tensor provided to _prepare_image_bytes.") # CORRECTED LOG
        return None, None

    print("[Fal Helper] Preparing image tensor...") # CORRECTED LOG
    try:
        if image_tensor.dim() == 4 and image_tensor.shape[0] == 1:
            img_tensor = image_tensor.squeeze(0)
        elif image_tensor.dim() == 3:
            img_tensor = image_tensor
        else:
            raise ValueError(f"Unexpected image tensor shape: {image_tensor.shape}")

        img_tensor = img_tensor.cpu()
        img_np = img_tensor.numpy()
        if img_np.max() <= 1.0 and img_np.min() >= 0.0:
             img_np = (img_np * 255)
        img_np = img_np.astype(np.uint8)
        pil_image = Image.fromarray(img_np, 'RGB')
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        print(f"[Fal Helper] Image tensor preparation complete ({len(img_bytes)} bytes).") # CORRECTED LOG
        return img_bytes, "image/png"

    except Exception as e:
        print(f"ERROR: [Fal Helper] Image tensor processing failed: {e}") # CORRECTED LOG
        traceback.print_exc()
        return None, None

def _save_tensor_to_temp_video(image_tensor_batch, fps=30):
    """Saves a ComfyUI Image Tensor Batch (B, H, W, C) to a temporary MP4 file."""
    if image_tensor_batch is None or image_tensor_batch.dim() != 4 or image_tensor_batch.shape[0] == 0:
        print("[Fal Helper] Invalid or empty image tensor batch provided for video saving.") # CORRECTED LOG
        return None

    print("[Fal Helper] Saving image tensor batch to temporary video file...") # CORRECTED LOG
    batch_size, height, width, channels = image_tensor_batch.shape
    if channels != 3:
        print(f"ERROR: [Fal Helper] Expected 3 color channels (RGB) for video saving, got {channels}.") # CORRECTED LOG
        return None

    output_dir = folder_paths.get_temp_directory()
    os.makedirs(output_dir, exist_ok=True)
    filename = f"fal_temp_upload_vid_{uuid.uuid4().hex}.mp4" # Generic name
    temp_video_filepath = os.path.join(output_dir, filename)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(temp_video_filepath, fourcc, float(fps), (width, height))

    if not video_writer.isOpened():
        print(f"ERROR: [Fal Helper] Failed to open video writer for path: {temp_video_filepath}") # CORRECTED LOG
        return None

    try:
        image_tensor_batch_cpu = image_tensor_batch.cpu()
        for i in range(batch_size):
            frame_tensor = image_tensor_batch_cpu[i]
            frame_np = frame_tensor.numpy()
            if frame_np.max() <= 1.0 and frame_np.min() >= 0.0:
                frame_np = (frame_np * 255)
            frame_np = frame_np.astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)

        print(f"[Fal Helper] Temporary video saved successfully to: {temp_video_filepath}") # CORRECTED LOG
        return temp_video_filepath

    except Exception as e:
        print(f"ERROR: [Fal Helper] Failed during video writing: {e}") # CORRECTED LOG
        traceback.print_exc()
        if video_writer.isOpened(): video_writer.release()
        if os.path.exists(temp_video_filepath):
            try: os.remove(temp_video_filepath)
            except Exception as clean_e: print(f"WARN: [Fal Helper] Failed cleanup on video write error: {clean_e}") # CORRECTED LOG
        return None
    finally:
        if video_writer.isOpened(): video_writer.release()

def _upload_media_to_fal(media_bytes, filename_hint, content_type):
    """
    Saves media bytes to a temporary file and uploads it using
    fal_client.upload_file, then returns the URL. Cleans up the temp file.
    """
    if not media_bytes:
        print(f"ERROR: [Fal Helper] No media bytes provided for upload ({filename_hint}).") # CORRECTED LOG
        return None

    temp_path = None
    try:
        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)
        ext = os.path.splitext(filename_hint)[1]
        if not ext and content_type:
            if 'png' in content_type: ext = '.png'
            elif 'jpeg' in content_type or 'jpg' in content_type: ext = '.jpg'
            elif 'mp4' in content_type: ext = '.mp4'
            elif 'webm' in content_type: ext = '.webm'
            elif 'mp3' in content_type or 'mpeg' in content_type: ext = '.mp3'
            elif 'wav' in content_type: ext = '.wav'
        if not ext: ext = ".tmp"

        temp_filename = f"fal_upload_{uuid.uuid4().hex}{ext}"
        temp_path = os.path.join(temp_dir, temp_filename)

        print(f"[Fal Helper] Writing temporary file for upload: {temp_path} ({len(media_bytes)} bytes)") # CORRECTED LOG
        with open(temp_path, "wb") as f: f.write(media_bytes)

        print(f"[Fal Helper] Uploading {temp_path} via fal_client.upload_file...") # CORRECTED LOG
        file_url = fal_client.upload_file(temp_path)
        print(f"[Fal Helper] Upload successful. URL: {file_url}") # CORRECTED LOG
        return file_url

    except Exception as e:
        print(f"ERROR: [Fal Helper] Fal.ai media upload failed for {filename_hint} (using temp file {temp_path}): {e}") # CORRECTED LOG
        traceback.print_exc()
        return None
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                print(f"[Fal Helper] Cleaning up temporary upload file: {temp_path}") # CORRECTED LOG
                os.remove(temp_path)
            except Exception as cleanup_e:
                print(f"WARN: [Fal Helper] Failed to delete temporary upload file '{temp_path}': {cleanup_e}") # CORRECTED LOG

def _save_audio_tensor_to_temp_wav(audio_data):
    """
    Saves audio data (from ComfyUI AUDIO type) to a temporary WAV file.
    Handles both 'samples' and 'waveform' keys for the tensor.
    Returns the path to the temporary file.
    """
    # --- Debug prints ---
    print(f"[Fal Helper] _save_audio_tensor_to_temp_wav received audio_data type: {type(audio_data)}") # Keep debug prefix
    if isinstance(audio_data, dict):
        print(f"[Fal Helper] audio_data keys: {audio_data.keys()}") # Keep debug prefix
    # --- End Debug ---

    if not isinstance(audio_data, dict) or 'sample_rate' not in audio_data or \
       ('samples' not in audio_data and 'waveform' not in audio_data):
        print(f"ERROR: [Fal Helper] Invalid audio data received. Expected dict with 'sample_rate' and either 'samples' or 'waveform'.") # CORRECTED LOG
        return None

    sample_rate = audio_data['sample_rate']
    samples_tensor = audio_data.get('samples') or audio_data.get('waveform')
    if samples_tensor is None:
         print(f"ERROR: [Fal Helper] Could not find audio tensor under 'samples' or 'waveform' key.") # CORRECTED LOG
         return None

    print(f"[Fal Helper] Processing audio tensor (Sample Rate: {sample_rate}, Shape: {samples_tensor.shape})") # CORRECTED LOG

    try:
        samples_tensor = samples_tensor.cpu()
        if samples_tensor.dim() == 3:
            print("[Fal Helper] Audio tensor has batch dimension, using first item.") # CORRECTED LOG
            samples_tensor = samples_tensor[0]
        elif samples_tensor.dim() != 2:
             raise ValueError(f"Unexpected audio tensor dimensions: {samples_tensor.shape}. Expected (channels, samples) or (batch, channels, samples).")

        samples_np = samples_tensor.numpy().T
        if np.issubdtype(samples_np.dtype, np.floating):
             print("[Fal Helper] Converting float audio to int16 for WAV export.") # CORRECTED LOG
             samples_np = np.clip(samples_np, -1.0, 1.0)
             samples_np = (samples_np * 32767).astype(np.int16)
        elif not np.issubdtype(samples_np.dtype, np.integer):
             print(f"WARN: [Fal Helper] Audio tensor is not float or integer type ({samples_np.dtype}). WAV export might be incorrect.") # CORRECTED LOG
             samples_np = samples_np.astype(np.int16)

        output_dir = folder_paths.get_temp_directory()
        os.makedirs(output_dir, exist_ok=True)
        filename = f"fal_temp_upload_aud_{uuid.uuid4().hex}.wav" # Generic name
        temp_audio_filepath = os.path.join(output_dir, filename)

        print(f"[Fal Helper] Saving temporary WAV file: {temp_audio_filepath}") # CORRECTED LOG
        scipy.io.wavfile.write(temp_audio_filepath, sample_rate, samples_np)
        return temp_audio_filepath

    except Exception as e:
        print(f"ERROR: [Fal Helper] Failed to save audio tensor to temporary WAV: {e}") # CORRECTED LOG
        traceback.print_exc()
        return None


# --- Define the Image-to-Video Node Class with Polling ---
class FalAPIVideoGeneratorI2V:
    # INPUT_TYPES, RETURN_TYPES, RETURN_NAMES, FUNCTION, CATEGORY remain the same
    @classmethod
    def INPUT_TYPES(cls): return {...} # Keep original definition
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_batch",)
    FUNCTION = "generate_video"
    CATEGORY = "BS_FalAi-API-Video/Image-to-Video"

    # _prepare_image method remains the same
    def _prepare_image(self, ...): ...

    # --- generate_video method for I2V with Polling ---
    def generate_video(self, model_name, api_key, seed, prompt,
                         image=None, negative_prompt=None,
                         resolution_enum="auto", aspect_ratio_enum="auto",
                         duration_seconds=5.0,
                         guidance_scale=7.5, steps=25,
                         num_frames=0,
                         style="auto",
                         prompt_optimizer=False, cleanup_temp_video=True):

        # Use a specific log prefix for this node
        def log_prefix(): return "FalAPIVideoGeneratorI2V:"

        # --- 1. Get Model Config ---
        # (Same as before)
        if model_name not in MODEL_CONFIGS["image_to_video"]:
            print(f"ERROR: {log_prefix()} Unknown model name '{model_name}'")
            return (None,)
        config = MODEL_CONFIGS["image_to_video"][model_name]
        endpoint_id = config['endpoint']
        expected_params = config['expected_params']
        print(f"{log_prefix()} Selected Model: {model_name}, Endpoint: {endpoint_id}")

        # --- 2. API Key Setup ---
        # (Same as before)
        if not api_key or not api_key.strip() or api_key == "Paste FAL_KEY credentials here (e.g., key_id:key_secret)":
            print(f"ERROR: {log_prefix()} API Key is missing.")
            return (None,)
        api_key_value = api_key.strip()
        # ... (format check, set os.environ) ...
        try:
            os.environ["FAL_KEY"] = api_key_value
            print(f"{log_prefix()} Using provided API Key.")
        except Exception as e:
             print(f"ERROR: {log_prefix()} Failed to set API Key environment variable: {e}")
             traceback.print_exc(); return (None,)


        # --- 3. Prepare Payload Dynamically ---
        # (Same payload construction logic as before, including image handling)
        payload = {}
        img_data_uri = None
        if 'image_url' in expected_params:
            if image is not None:
                # Use internal _prepare_image if available, or fallback/error
                if hasattr(self, '_prepare_image'):
                    img_data_uri = self._prepare_image(image)
                else: # Or use _prepare_image_bytes if _prepare_image wasn't defined/copied
                    img_bytes, _ = _prepare_image_bytes(image)
                    if img_bytes: img_data_uri = f"data:image/png;base64,{base64.b64encode(img_bytes).decode('utf-8')}"

                if img_data_uri: payload['image_url'] = img_data_uri
                else:
                    print(f"ERROR: {log_prefix()} Failed to prepare image.")
                    return (None,)
            else: print(f"WARN: {log_prefix()} Model expects 'image_url', but none provided.")
        elif image is not None: print(f"WARN: {log_prefix()} Image provided but model doesn't expect 'image_url'.")
        # ... (rest of payload construction: prompts, numeric params, enums) ...
        # (Ensure this part correctly builds the 'payload' dictionary)
        param_map = {...} # As before
        api_name_map = {...} # As before
        for input_name, value in param_map.items():
            # ... (Logic to add value to payload if expected and valid) ...
            api_name = api_name_map.get(input_name, input_name)
            if api_name in expected_params:
                 # ... (Handle 'auto', type conversions, add to payload) ...
                 pass # Placeholder for existing logic
        # ... (Handle resolution/aspect ratio enums) ...


        # --- 4. API Call with Polling, Download, Frame Extraction ---
        request_id = None
        video_url = None
        temp_video_filepath = None
        frames_tensor = None # Initialize

        try:
            print(f"{log_prefix()} Submitting job to endpoint: {endpoint_id}")
            if not endpoint_id or not endpoint_id.strip():
                 raise ValueError("Endpoint ID cannot be empty.")
            print(f"{log_prefix()} Payload: {json.dumps(payload, indent=2)}") # Debug payload

            # --- Submit and Poll ---
            handler = fal_client.submit(endpoint_id.strip(), arguments=payload)
            request_id = handler.request_id
            print(f"{log_prefix()} Job submitted. Request ID: {request_id}")

            # Poll for the result (e.g., 15 min timeout = 900s)
            response = _poll_fal_job(endpoint_id, request_id, timeout=900)
            print(f"{log_prefix()} Job {request_id} completed successfully (poller returned).")
            # 'response' now holds the final result dictionary
            # --- End Submit and Poll ---


            # --- Process Response ---
            # (Same logic as before to find video_url from response dict)
            video_data = response.get('video')
            if isinstance(video_data, dict): video_url = video_data.get('url')
            elif isinstance(response.get('videos'), list) and len(response['videos']) > 0:
                 vid_info = response['videos'][0]
                 if isinstance(vid_info, dict): video_url = vid_info.get('url')
            elif isinstance(response, dict) and response.get('url') and isinstance(response.get('url'), str):
                 if any(ext in response['url'].lower() for ext in ['.mp4', '.webm', '.mov', '.avi']):
                      video_url = response['url']

            if not video_url:
                 print(f"ERROR: {log_prefix()} Polling succeeded but could not find video 'url' in Fal.ai result.")
                 print(f"--- Full result: {json.dumps(response, indent=2)}")
                 return (None,) # Correct return for this node

            print(f"{log_prefix()} Video URL received: {video_url}")

            # --- Download Video ---
            print(f"{log_prefix()} Downloading video file...")
            video_response = requests.get(video_url, stream=True, timeout=300)
            video_response.raise_for_status()

            # --- Save Temp File ---
            output_dir = folder_paths.get_temp_directory()
            os.makedirs(output_dir, exist_ok=True)
            content_type = video_response.headers.get('content-type', '').lower()
            extension = '.mp4' # Default
            if 'webm' in content_type: extension = '.webm'
            elif video_url.lower().endswith('.webm'): extension = '.webm'
            filename = f"fal_api_i2v_temp_{uuid.uuid4().hex}{extension}"
            temp_video_filepath = os.path.join(output_dir, filename)
            with open(temp_video_filepath, 'wb') as video_file:
                for chunk in video_response.iter_content(chunk_size=1024*1024): video_file.write(chunk)
            print(f"{log_prefix()} Video downloaded to: {temp_video_filepath}")

            # --- Extract Frames ---
            print(f"{log_prefix()} Extracting frames...")
            frames_list = []
            cap = cv2.VideoCapture(temp_video_filepath)
            if not cap.isOpened(): raise IOError(f"Could not open video file: {temp_video_filepath}")
            while True:
                ret, frame = cap.read()
                if not ret: break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_list.append(frame_rgb)
            cap.release()
            if not frames_list: raise ValueError(f"No frames extracted from video: {temp_video_filepath}")
            print(f"{log_prefix()} Extracted {len(frames_list)} frames.")

            frames_np = np.stack(frames_list, axis=0)
            frames_tensor = torch.from_numpy(frames_np).float() / 255.0
            # --- End Frame Extraction ---

            if frames_tensor is None: raise ValueError("Frame tensor processing failed.")
            print(f"{log_prefix()} Frames tensor shape: {frames_tensor.shape}")
            return (frames_tensor,) # Success return

        # --- Exception Handling with Poller ---
        except KeyboardInterrupt:
            print(f"ERROR: {log_prefix()} Execution interrupted by user.")
            if request_id:
                 print(f"{log_prefix()} Attempting to cancel Fal.ai job {request_id}...")
                 try: fal_client.cancel(endpoint_id, request_id)
                 except Exception as cancel_e: print(f"WARN: {log_prefix()} Failed to send cancel request: {cancel_e}")
            return (None,) # Correct return for this node
        except TimeoutError as e:
            print(f"ERROR: {log_prefix()} Job timed out: {e}")
            if request_id:
                 print(f"{log_prefix()} Attempting to cancel Fal.ai job {request_id} due to timeout...")
                 try: fal_client.cancel(endpoint_id, request_id)
                 except Exception as cancel_e: print(f"WARN: {log_prefix()} Failed to send cancel request after timeout: {cancel_e}")
            return (None,) # Correct return for this node
        except RuntimeError as e: # Fal job failures
             print(f"ERROR: {log_prefix()} Fal.ai job failed: {e}")
             return (None,) # Correct return for this node
        except requests.exceptions.RequestException as e:
             print(f"ERROR: {log_prefix()} Network request failed: {e}")
             traceback.print_exc(); return (None,)
        except (cv2.error, IOError, ValueError, Image.UnidentifiedImageError) as e: # Added Image error
             print(f"ERROR: {log_prefix()} Media processing error: {e}")
             traceback.print_exc(); return (None,)
        except Exception as e:
            req_id_str = f"Request ID: {request_id}" if request_id else 'N/A'
            print(f"ERROR: {log_prefix()} Unexpected error ({req_id_str}): {e}")
            traceback.print_exc(); return (None,)

        # --- Final Cleanup ---
        finally:
            # (Same cleanup logic as before)
            if cleanup_temp_video and temp_video_filepath and os.path.exists(temp_video_filepath):
                try:
                    print(f"{log_prefix()} Cleaning up temporary video file: {temp_video_filepath}")
                    os.remove(temp_video_filepath)
                except Exception as e:
                    print(f"WARN: {log_prefix()} Failed to delete temp file '{temp_video_filepath}': {e}")
            elif temp_video_filepath and os.path.exists(temp_video_filepath):
                 print(f"{log_prefix()} Keeping temporary video file: {temp_video_filepath}")


# --- Define the Text-to-Video Node Class with Polling ---
class FalAPIVideoGeneratorT2V:
    # INPUT_TYPES, RETURN_TYPES, RETURN_NAMES, FUNCTION, CATEGORY remain the same
    @classmethod
    def INPUT_TYPES(cls): return {...} # Keep original definition
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_batch",)
    FUNCTION = "generate_video"
    CATEGORY = "BS_FalAi-API-Video/Text-to-Video"

    # NO _prepare_image method needed

    # --- generate_video method for T2V with Polling ---
    def generate_video(self, model_name, api_key, seed, prompt,
                         negative_prompt=None,
                         resolution_enum="auto", aspect_ratio_enum="auto",
                         duration_seconds=5.0,
                         guidance_scale=7.5, steps=25,
                         num_frames=0,
                         style="auto",
                         prompt_optimizer=False, cleanup_temp_video=True):

        def log_prefix(): return "FalAPIVideoGeneratorT2V:" # Specific prefix

        # --- 1. Get Model Config ---
        # (Same as before)
        if model_name not in MODEL_CONFIGS["text_to_video"]:
            print(f"ERROR: {log_prefix()} Unknown model name '{model_name}'")
            return (None,)
        config = MODEL_CONFIGS["text_to_video"][model_name]
        endpoint_id = config['endpoint']
        expected_params = config['expected_params']
        print(f"{log_prefix()} Selected Model: {model_name}, Endpoint: {endpoint_id}")

        # --- 2. API Key Setup ---
        # (Same as before)
        if not api_key or not api_key.strip() or api_key == "Paste FAL_KEY credentials here (e.g., key_id:key_secret)":
            print(f"ERROR: {log_prefix()} API Key is missing.")
            return (None,)
        api_key_value = api_key.strip()
        try:
            os.environ["FAL_KEY"] = api_key_value
            print(f"{log_prefix()} Using provided API Key.")
        except Exception as e:
             print(f"ERROR: {log_prefix()} Failed to set API Key environment variable: {e}")
             traceback.print_exc(); return (None,)

        # --- 3. Prepare Payload Dynamically ---
        # (Same payload construction logic as before, NO image handling)
        payload = {}
        # ... (Add prompt, negative_prompt, numeric, enum params to payload) ...
        # (Ensure this part correctly builds the 'payload' dictionary)
        param_map = {...} # As before
        api_name_map = {...} # As before
        for input_name, value in param_map.items():
            # ... (Logic to add value to payload if expected and valid) ...
            api_name = api_name_map.get(input_name, input_name)
            if api_name in expected_params:
                 # ... (Handle 'auto', type conversions, add to payload) ...
                 pass # Placeholder for existing logic
        # ... (Handle resolution/aspect ratio enums) ...

        # --- 4. API Call with Polling, Download, Frame Extraction ---
        request_id = None
        video_url = None
        temp_video_filepath = None
        frames_tensor = None # Initialize

        try:
            print(f"{log_prefix()} Submitting job to endpoint: {endpoint_id}")
            if not endpoint_id or not endpoint_id.strip():
                 raise ValueError("Endpoint ID cannot be empty.")
            print(f"{log_prefix()} Payload: {json.dumps(payload, indent=2)}") # Debug payload

            # --- Submit and Poll ---
            handler = fal_client.submit(endpoint_id.strip(), arguments=payload)
            request_id = handler.request_id
            print(f"{log_prefix()} Job submitted. Request ID: {request_id}")

            # Poll for the result (e.g., 15 min timeout = 900s)
            response = _poll_fal_job(endpoint_id, request_id, timeout=900)
            print(f"{log_prefix()} Job {request_id} completed successfully (poller returned).")
            # --- End Submit and Poll ---

            # --- Process Response ---
            # (Same logic as I2V to find video_url from response dict)
            video_data = response.get('video')
            if isinstance(video_data, dict): video_url = video_data.get('url')
            # ... (other checks for video url) ...
            if not video_url:
                 print(f"ERROR: {log_prefix()} Polling succeeded but could not find video 'url' in Fal.ai result.")
                 print(f"--- Full result: {json.dumps(response, indent=2)}")
                 return (None,) # Correct return

            print(f"{log_prefix()} Video URL received: {video_url}")

            # --- Download, Save, Extract ---
            # (Same logic as I2V node)
            print(f"{log_prefix()} Downloading video file...")
            video_response = requests.get(video_url, stream=True, timeout=300)
            video_response.raise_for_status()
            output_dir = folder_paths.get_temp_directory()
            os.makedirs(output_dir, exist_ok=True)
            # ... (determine extension) ...
            filename = f"fal_api_t2v_temp_{uuid.uuid4().hex}{extension}" # Use T2V prefix
            temp_video_filepath = os.path.join(output_dir, filename)
            with open(temp_video_filepath, 'wb') as video_file: # ... (write file) ...
            print(f"{log_prefix()} Video downloaded to: {temp_video_filepath}")
            # ... (extract frames into frames_list) ...
            if not frames_list: raise ValueError(f"No frames extracted from video: {temp_video_filepath}")
            print(f"{log_prefix()} Extracted {len(frames_list)} frames.")
            frames_np = np.stack(frames_list, axis=0)
            frames_tensor = torch.from_numpy(frames_np).float() / 255.0

            if frames_tensor is None: raise ValueError("Frame tensor processing failed.")
            print(f"{log_prefix()} Frames tensor shape: {frames_tensor.shape}")
            return (frames_tensor,) # Success return

        # --- Exception Handling with Poller ---
        except KeyboardInterrupt:
            print(f"ERROR: {log_prefix()} Execution interrupted by user.")
            if request_id:
                 print(f"{log_prefix()} Attempting to cancel Fal.ai job {request_id}...")
                 try: fal_client.cancel(endpoint_id, request_id)
                 except Exception as cancel_e: print(f"WARN: {log_prefix()} Failed to send cancel request: {cancel_e}")
            return (None,) # Correct return for this node
        except TimeoutError as e:
            print(f"ERROR: {log_prefix()} Job timed out: {e}")
            if request_id:
                 print(f"{log_prefix()} Attempting to cancel Fal.ai job {request_id} due to timeout...")
                 try: fal_client.cancel(endpoint_id, request_id)
                 except Exception as cancel_e: print(f"WARN: {log_prefix()} Failed to send cancel request after timeout: {cancel_e}")
            return (None,) # Correct return for this node
        except RuntimeError as e: # Fal job failures
             print(f"ERROR: {log_prefix()} Fal.ai job failed: {e}")
             return (None,) # Correct return for this node
        except requests.exceptions.RequestException as e:
             print(f"ERROR: {log_prefix()} Network request failed: {e}")
             traceback.print_exc(); return (None,)
        except (cv2.error, IOError, ValueError, Image.UnidentifiedImageError) as e:
             print(f"ERROR: {log_prefix()} Media processing error: {e}")
             traceback.print_exc(); return (None,)
        except Exception as e:
            req_id_str = f"Request ID: {request_id}" if request_id else 'N/A'
            print(f"ERROR: {log_prefix()} Unexpected error ({req_id_str}): {e}")
            traceback.print_exc(); return (None,)

        # --- Final Cleanup ---
        finally:
            # (Same cleanup logic as before)
            if cleanup_temp_video and temp_video_filepath and os.path.exists(temp_video_filepath):
                # ... (try os.remove) ...
                pass # Placeholder for existing logic
            elif temp_video_filepath and os.path.exists(temp_video_filepath):
                 print(f"{log_prefix()} Keeping temporary video file: {temp_video_filepath}")


# --- Define the Omni Pro Node Class with Polling ---
class FalAPIOmniProNode:
    # Constants, INPUT_TYPES, RETURN_TYPES, RETURN_NAMES, FUNCTION, CATEGORY remain the same
    AUTO_KEY_START_IMAGE = "image_url"
    AUTO_KEY_END_IMAGE = "end_image_url"
    AUTO_KEY_INPUT_VIDEO = "video_url"
    AUTO_KEY_INPUT_AUDIO = "audio_url"
    @classmethod
    def INPUT_TYPES(cls): return {...} # Keep original definition
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_batch",)
    FUNCTION = "execute_omni_request"
    CATEGORY = "BS_FalAi-API-Omni"

    # --- execute_omni_request method with Polling ---
    def execute_omni_request(self, endpoint_id, api_key, parameters_json,
                             start_image=None, end_image=None, input_video=None,
                             input_audio=None,
                             cleanup_temp_files=True,
                             output_video_fps=30):

        def log_prefix(): return "FalAPIOmniProNode:" # Specific prefix

        # --- 1. API Key Setup ---
        # (Same as before)
        if not api_key or not api_key.strip() or api_key == "Paste FAL_KEY credentials here (e.g., key_id:key_secret)":
            print(f"ERROR: {log_prefix()} API Key is missing.")
            return (None,) # Correct return
        api_key_value = api_key.strip()
        try: os.environ["FAL_KEY"] = api_key_value; print(f"{log_prefix()} Using provided API Key.")
        except Exception as e: print(f"ERROR: {log_prefix()} Failed to set API Key: {e}"); traceback.print_exc(); return (None,) # Correct return

        # --- 2. Parse User Parameters JSON ---
        # (Same as before)
        user_params = {}
        try:
             if parameters_json and parameters_json.strip(): user_params = json.loads(parameters_json); # ... (validate dict) ...
        except Exception as e: print(f"ERROR: {log_prefix()} Invalid JSON: {e}"); return (None,) # Correct return

        # --- 3. Handle and Upload Media Inputs (Automatic) ---
        # (Same logic as before using helper functions _prepare_image_bytes, _save_tensor_to_temp_video, _save_audio_tensor_to_temp_wav, _upload_media_to_fal)
        uploaded_media_urls = {}
        temp_files_to_clean = []
        upload_error = False
        try:
            # ... (Upload start_image if present) ...
            if start_image is not None: # ... (prepare, upload, handle error) ...
                 pass # Placeholder for existing logic
            # ... (Upload end_image if present) ...
            if end_image is not None and not upload_error: # ... (prepare, upload, handle error) ...
                 pass # Placeholder for existing logic
            # ... (Upload input_video if present) ...
            if input_video is not None and not upload_error: # ... (save, read, upload, handle error) ...
                 pass # Placeholder for existing logic
            # ... (Upload input_audio if present) ...
            if input_audio is not None and not upload_error: # ... (save, read, upload, handle error) ...
                 pass # Placeholder for existing logic
        except Exception as e: print(f"ERROR: {log_prefix()} Media processing error: {e}"); upload_error = True
        if upload_error: print(f"ERROR: {log_prefix()} Aborting due to media errors."); # ... (cleanup) ...
            return (None,) # Correct return

        # --- 4. Construct Final Payload ---
        # (Same logic as before)
        final_payload = user_params.copy()
        for auto_key, url in uploaded_media_urls.items(): # ... (inject/overwrite) ...
            pass # Placeholder for existing logic
        print(f"{log_prefix()} Final Payload: {json.dumps(final_payload, indent=2)}") # Debug payload

        # --- 5. API Call with Polling, Download, Frame Extraction ---
        request_id = None
        result_url = None
        result_content_type = None
        temp_download_filepath = None
        frames_tensor = None # Initialize for video case
        img_tensor = None # Initialize for image case

        try:
            print(f"{log_prefix()} Submitting job to endpoint: {endpoint_id}")
            if not endpoint_id or not endpoint_id.strip(): raise ValueError("Endpoint ID cannot be empty.")

            # --- Submit and Poll ---
            handler = fal_client.submit(endpoint_id.strip(), arguments=final_payload)
            request_id = handler.request_id
            print(f"{log_prefix()} Job submitted. Request ID: {request_id}")

            response = _poll_fal_job(endpoint_id, request_id, timeout=900) # Adjust timeout as needed
            print(f"{log_prefix()} Job {request_id} completed successfully (poller returned).")
            # --- End Submit and Poll ---

            # --- Process Response (Flexible Omni logic) ---
            # (Same flexible logic as before to find result_url and determine if video/image)
            is_video = False
            is_image = False
            # ... (logic to set result_url, result_content_type, is_video, is_image) ...
            # Ensure this section correctly identifies the media type and URL from 'response'
            if not result_url: print(f"WARN: {log_prefix()} No media URL found in result."); return (None,) # Correct return

            # --- Download and Process Result ---
            print(f"{log_prefix()} Downloading result: {result_url} (Type: {'Video' if is_video else 'Image' if is_image else 'Unknown'})")
            media_response = requests.get(result_url, stream=True, timeout=600)
            media_response.raise_for_status()

            if is_video:
                # (Same logic as I2V/T2V to save temp file, extract frames, create frames_tensor)
                # ... (save to temp_download_filepath) ...
                # ... (extract frames into frames_list) ...
                if not frames_list: raise ValueError("No frames extracted")
                frames_np = np.stack(frames_list, axis=0)
                frames_tensor = torch.from_numpy(frames_np).float() / 255.0
                if frames_tensor is None: raise ValueError("Frame tensor processing failed.")
                print(f"{log_prefix()} Video result processed. Tensor shape: {frames_tensor.shape}")
                return (frames_tensor,) # Success return for video

            elif is_image:
                # (Same logic as Omni node before to process image)
                image_bytes = media_response.content
                pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                img_np = np.array(pil_image, dtype=np.float32) / 255.0
                img_tensor = torch.from_numpy(img_np).unsqueeze(0)
                print(f"{log_prefix()} Image result processed. Tensor shape: {img_tensor.shape}")
                return (img_tensor,) # Success return for image

            else:
                 print(f"ERROR: {log_prefix()} Could not determine result type (video/image).")
                 return (None,) # Correct return

        # --- Exception Handling with Poller ---
        except KeyboardInterrupt:
            print(f"ERROR: {log_prefix()} Execution interrupted by user.")
            if request_id: # ... (attempt cancel) ...
                pass # Placeholder for cancel logic
            return (None,) # Correct return for this node
        except TimeoutError as e:
            print(f"ERROR: {log_prefix()} Job timed out: {e}")
            if request_id: # ... (attempt cancel) ...
                pass # Placeholder for cancel logic
            return (None,) # Correct return for this node
        except RuntimeError as e: # Fal job failures
             print(f"ERROR: {log_prefix()} Fal.ai job failed: {e}")
             return (None,) # Correct return for this node
        except requests.exceptions.RequestException as e:
             print(f"ERROR: {log_prefix()} Network request failed: {e}")
             traceback.print_exc(); return (None,)
        except (cv2.error, IOError, ValueError, Image.UnidentifiedImageError) as e:
             print(f"ERROR: {log_prefix()} Media processing error: {e}")
             traceback.print_exc(); return (None,)
        except Exception as e:
            req_id_str = f"Request ID: {request_id}" if request_id else 'N/A'
            print(f"ERROR: {log_prefix()} Unexpected error ({req_id_str}): {e}")
            traceback.print_exc(); return (None,)

        # --- Final Cleanup ---
        finally:
            # (Same cleanup logic as before)
            if cleanup_temp_files: # ... (cleanup temp_files_to_clean and temp_download_filepath if exists) ...
                pass # Placeholder for cleanup logic
            else: print(f"{log_prefix()} Skipping cleanup.")


# --- Define the LipSync Node Class with Polling ---
class FalAILipSyncNode:
    # Constants, INPUT_TYPES, RETURN_TYPES, RETURN_NAMES, FUNCTION, CATEGORY remain the same
    PAYLOAD_KEY_VIDEO = "video_url"
    PAYLOAD_KEY_AUDIO = "audio_url"
    PAYLOAD_KEY_SYNC_MODE = "sync_mode"
    PAYLOAD_KEY_MODEL = "model"
    ENDPOINT_V2 = "fal-ai/sync-lipsync/v2"
    ENDPOINT_V1_9 = "fal-ai/sync-lipsync"
    MODEL_OPTIONS_V1_9 = ["lipsync-1.9.0-beta", "lipsync-1.8.0", "lipsync-1.7.1"]
    @classmethod
    def INPUT_TYPES(cls): return {...} # Keep original definition
    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("image_batch", "synced_audio", "video_path")
    FUNCTION = "execute_lipsync"
    CATEGORY = "BS_FalAi-API-Video"

    # --- execute_lipsync method with Polling ---
    def execute_lipsync(self, api_key, endpoint_version, input_video, input_audio,
                        sync_mode="cut_off", model="lipsync-1.9.0-beta",
                        cleanup_temp_files=True,
                        output_video_fps=30):

        def log_prefix(): return "FalAILipSyncNode:" # Specific prefix

        # --- Setup and Initializations ---
        print(f"{log_prefix()} Starting LipSync request execution for version: {endpoint_version}")
        uploaded_media_urls = {}
        temp_files_to_clean = []
        payload = {}
        temp_download_filepath = None
        frames_tensor = None # Initialize here

        # --- 1. Select Endpoint ---
        if endpoint_version == "v2.0": endpoint_to_call = self.ENDPOINT_V2
        elif endpoint_version == "v1.9": endpoint_to_call = self.ENDPOINT_V1_9
        else: print(f"ERROR: {log_prefix()} Invalid endpoint_version"); return (None, None, None)
        print(f"{log_prefix()} Targeting endpoint: {endpoint_to_call}")

        # --- 2. API Key Setup ---
        if not api_key or not api_key.strip() or api_key == "Paste FAL_KEY credentials here (e.g., key_id:key_secret)":
            print(f"ERROR: {log_prefix()} API Key is missing."); return (None, None, None)
        api_key_value = api_key.strip()
        try: os.environ["FAL_KEY"] = api_key_value; print(f"{log_prefix()} Using provided API Key.")
        except Exception as e: print(f"ERROR: {log_prefix()} Failed to set API Key: {e}"); traceback.print_exc(); return (None, None, None)

        # --- 3. Validate Required Inputs ---
        if input_video is None: print(f"ERROR: {log_prefix()} 'input_video' is required."); return (None, None, None)
        if input_audio is None: print(f"ERROR: {log_prefix()} 'input_audio' is required."); return (None, None, None)

        # --- 4. Handle and Upload Media Inputs ---
        upload_error = False
        try:
            # Video Upload
            print(f"{log_prefix()} Processing required input_video...")
            temp_video_path = _save_tensor_to_temp_video(input_video, fps=output_video_fps)
            if temp_video_path and os.path.exists(temp_video_path):
                temp_files_to_clean.append(temp_video_path)
                with open(temp_video_path, 'rb') as vf: video_bytes = vf.read()
                if video_bytes:
                    url = _upload_media_to_fal(video_bytes, "input_video.mp4", "video/mp4")
                    if url: uploaded_media_urls[self.PAYLOAD_KEY_VIDEO] = url
                    else: upload_error = True; print(f"ERROR: {log_prefix()} Failed video upload.")
                else: upload_error = True; print(f"ERROR: {log_prefix()} Failed reading temp video bytes.")
            else: upload_error = True; print(f"ERROR: {log_prefix()} Failed saving video tensor.")
            # Audio Upload
            if not upload_error:
                print(f"{log_prefix()} Processing required input_audio...")
                temp_audio_path = _save_audio_tensor_to_temp_wav(input_audio)
                if temp_audio_path and os.path.exists(temp_audio_path):
                    temp_files_to_clean.append(temp_audio_path)
                    with open(temp_audio_path, 'rb') as af: audio_bytes = af.read()
                    if audio_bytes:
                        url = _upload_media_to_fal(audio_bytes, "input_audio.wav", "audio/wav")
                        if url: uploaded_media_urls[self.PAYLOAD_KEY_AUDIO] = url
                        else: upload_error = True; print(f"ERROR: {log_prefix()} Failed audio upload.")
                    else: upload_error = True; print(f"ERROR: {log_prefix()} Failed reading temp audio bytes.")
                else: upload_error = True; print(f"ERROR: {log_prefix()} Failed saving audio tensor.")
        except Exception as e: print(f"ERROR: {log_prefix()} Media processing error: {e}"); traceback.print_exc(); upload_error = True
        if upload_error: print(f"ERROR: {log_prefix()} Aborting due to media errors."); # ... (cleanup) ...
            return (None, None, None)

        # --- 5. Construct Final Payload ---
        payload = {}
        payload[self.PAYLOAD_KEY_VIDEO] = uploaded_media_urls[self.PAYLOAD_KEY_VIDEO]
        payload[self.PAYLOAD_KEY_AUDIO] = uploaded_media_urls[self.PAYLOAD_KEY_AUDIO]
        payload[self.PAYLOAD_KEY_SYNC_MODE] = sync_mode
        if endpoint_version == "v1.9": payload[self.PAYLOAD_KEY_MODEL] = model; print(f"{log_prefix()} Adding 'model': {model} for v1.9.")
        elif model != self.MODEL_OPTIONS_V1_9[0]: print(f"WARN: {log_prefix()} 'model' parameter ('{model}') ignored for v2.0.")
        print(f"{log_prefix()} Final Payload: {json.dumps(payload, indent=2)}")

        # --- 6. API Call with Polling, Download, Frame Extraction ---
        request_id = None
        result_url = None
        result_content_type = None
        # frames_tensor = None # Already initialized

        try:
            print(f"{log_prefix()} Submitting job to: {endpoint_to_call}")
            # --- Submit and Poll ---
            handler = fal_client.submit(endpoint_to_call, arguments=payload)
            request_id = handler.request_id
            print(f"{log_prefix()} Job submitted. Request ID: {request_id}")
            response = _poll_fal_job(endpoint_to_call, request_id, timeout=900) # Adjust timeout
            print(f"{log_prefix()} Job {request_id} completed successfully (poller returned).")
            # --- End Submit and Poll ---

            # --- Process Response ---
            if isinstance(response, dict) and 'video' in response and isinstance(response['video'], dict) and 'url' in response['video']:
                result_url = response['video']['url']
                result_content_type = response['video'].get('content_type', 'video/mp4')
            else:
                 print(f"ERROR: {log_prefix()} Polling succeeded but couldn't find 'video.url' in result.")
                 print(f"--- Final result: --- \n{json.dumps(response, indent=2)}")
                 return (None, None, None) # Correct return

            print(f"{log_prefix()} Result Video URL: {result_url}")

            # --- Download, Save, Extract ---
            print(f"{log_prefix()} Downloading result video...")
            media_response = requests.get(result_url, stream=True, timeout=600)
            media_response.raise_for_status()
            output_dir = folder_paths.get_temp_directory()
            os.makedirs(output_dir, exist_ok=True)
            extension = '.mp4'
            if result_content_type and 'webm' in result_content_type: extension = '.webm'
            elif result_url.lower().endswith('.webm'): extension = '.webm'
            filename = f"fal_lipsync_result_{uuid.uuid4().hex}{extension}"
            temp_download_filepath = os.path.join(output_dir, filename)
            temp_files_to_clean.append(temp_download_filepath)
            with open(temp_download_filepath, 'wb') as f_out:
                for chunk in media_response.iter_content(chunk_size=1024*1024): f_out.write(chunk)
            print(f"{log_prefix()} Video downloaded to: {temp_download_filepath}")

            print(f"{log_prefix()} Extracting frames...")
            frames_list = []
            cap = cv2.VideoCapture(temp_download_filepath)
            if not cap.isOpened(): raise IOError(f"Could not open downloaded video: {temp_download_filepath}")
            while True: ret, frame = cap.read(); # ... (extract loop) ...
                 if not ret: break; frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); frames_list.append(frame_rgb) # Condensed loop
            cap.release()
            if not frames_list: raise ValueError(f"No frames extracted from video: {temp_download_filepath}")
            print(f"{log_prefix()} Extracted {len(frames_list)} frames.")
            frames_np = np.stack(frames_list, axis=0)
            frames_tensor = torch.from_numpy(frames_np).float() / 255.0

            if frames_tensor is None: raise ValueError("Frame tensor processing failed.") # Keep check
            print(f"{log_prefix()} Frames tensor shape: {frames_tensor.shape}")
            return (frames_tensor, input_audio, temp_download_filepath) # Success return

        # --- Exception Handling with Poller ---
        except KeyboardInterrupt:
            print(f"ERROR: {log_prefix()} Execution interrupted by user.")
            if request_id: print(f"{log_prefix()} Attempting cancel..."); # ... (try fal_client.cancel) ...
            return (None, None, None) # Correct return
        except TimeoutError as e:
            print(f"ERROR: {log_prefix()} Job timed out: {e}")
            if request_id: print(f"{log_prefix()} Attempting cancel..."); # ... (try fal_client.cancel) ...
            return (None, None, None) # Correct return
        except RuntimeError as e: # Fal job failures
             print(f"ERROR: {log_prefix()} Fal.ai job failed: {e}")
             return (None, None, None) # Correct return
        except requests.exceptions.RequestException as e:
             print(f"ERROR: {log_prefix()} Network request failed: {e}")
             traceback.print_exc(); return (None, None, None)
        except (cv2.error, IOError, ValueError, Image.UnidentifiedImageError) as e:
             print(f"ERROR: {log_prefix()} Media processing error: {e}")
             traceback.print_exc(); return (None, None, None)
        except Exception as e:
            req_id_str = f"Request ID: {request_id}" if request_id else 'N/A'
            print(f"ERROR: {log_prefix()} Unexpected error ({req_id_str}): {e}")
            traceback.print_exc(); return (None, None, None)

        # --- Final Cleanup ---
        finally:
            # (Same cleanup logic as before)
            if cleanup_temp_files: # ... (loop and remove temp_files_to_clean) ...
                pass # Placeholder
            else: # ... (print skipping message) ...
                pass # Placeholder


# --- Node Registration (Keep as is) ---
NODE_CLASS_MAPPINGS = {
    "FalAPIVideoGeneratorI2V": FalAPIVideoGeneratorI2V,
    "FalAPIVideoGeneratorT2V": FalAPIVideoGeneratorT2V,
    "FalAPIOmniProNode": FalAPIOmniProNode,
    "FalAILipSyncNode": FalAILipSyncNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FalAPIVideoGeneratorI2V": "FAL AI Image-to-Video",
    "FalAPIVideoGeneratorT2V": "FAL AI Text-to-Video",
    "FalAPIOmniProNode": "FAL AI API Omni Pro Node",
    "FalAILipSyncNode": "FAL AI API LipSync Node (v1.9/v2.0)",
}
