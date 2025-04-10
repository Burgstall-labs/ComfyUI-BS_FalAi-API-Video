# Full nodes.py content with corrected indentation and polling logic

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
# (MODEL_CONFIGS dictionary remains the same)
MODEL_CONFIGS = {
    "image_to_video": {
        "MiniMax (Hailuo AI) Video 01 Image to Video": {
            "endpoint": "fal-ai/minimax/video-01/image-to-video",
            "resolutions": [], "aspect_ratios": [], "durations": [],
            "schema_str": "[Prompt:String], [Image_url:String]",
        },
        "Kling 1.6 Image to Video (Pro)": {
            "endpoint": "fal-ai/kling-video/v1.6/pro/image-to-video",
            "resolutions": [], "aspect_ratios": ["16:9", "9:16", "1:1"], "durations": [5, 10],
            "schema_str": "[Prompt:String], [Image_url:String], [duration:DurationEnum], [aspect_ratio:AspectRatioEnum]",
        },
        "Veo 2 (Image to Video) Image to Video": {
            "endpoint": "fal-ai/veo2/image-to-video",
            "resolutions": [], "aspect_ratios": ["auto", "auto_prefer_portrait", "16:9", "9:16"], "durations": [5, 6, 7, 8],
            "schema_str": "[Prompt:String], [Image_url:String], [aspect_ratio:AspectRatioEnum], [duration:DurationEnum]",
        },
        "PixVerse v4: Image to Video Fast Image to Video": {
            "endpoint": "fal-ai/pixverse/v4/image-to-video/fast",
            "resolutions": ["360p", "540p", "720p"], "aspect_ratios": ["16:9", "4:3", "1:1", "3:4", "9:16"], "durations": [],
            "schema_str": "[Prompt:String], [aspect_ratio:AspectRatioEnum], [resolution:ResolutionEnum], [negative_prompt:String], [style:Enum], [seed:Integer], [image_url:String]",
        },
        "PixVerse v4: Image to Video Image to Video": {
            "endpoint": "fal-ai/pixverse/v4/image-to-video",
            "resolutions": ["360p", "540p", "720p", "1080p"], "aspect_ratios": ["16:9", "4:3", "1:1", "3:4", "9:16"], "durations": [5, 8],
            "schema_str": "[Prompt:String], [aspect_ratio:AspectRatioEnum], [resolution:ResolutionEnum], [duration:DurationEnum], [negative_prompt:String], [style:Enum], [seed:Integer], [image_url:String]",
        },
        "Luma Ray 2 Flash (Image to Video) Image to Video": {
            "endpoint": "fal-ai/luma-dream-machine/ray-2-flash/image-to-video",
            "resolutions": ["540p", "720p", "1080p"], "aspect_ratios": ["16:9", "9:16", "4:3", "3:4", "21:9", "9:21"], "durations": [5],
            "schema_str": "[Prompt:String], [image_url:String], [end_image_url:String], [aspect_ratio:AspectRatioEnum], [loop:Boolean], [resolution:ResolutionEnum], [duration:DurationEnum]",
        },
        "Pika Image to Video Turbo (v2) Image to Video": {
            "endpoint": "fal-ai/pika/v2/turbo/image-to-video",
            "resolutions": ["720p", "1080p"], "aspect_ratios": ["16:9", "9:16", "1:1", "4:5", "5:4", "3:2", "2:3"], "durations": [],
            "schema_str": "[Image_url:String], [Prompt:String], [seed:Integer], [negative_prompt:String], [resolution:ResolutionEnum], [duration:Integer]",
        },
        "Wan-2.1 Image-to-Video Image to Video": {
            "endpoint": "fal-ai/wan-i2v",
            "resolutions": ["480p", "720p"], "aspect_ratios": ["auto", "16:9", "9:16", "1:1"], "durations": [5],
            "schema_str": "[Prompt:String], [negative_prompt:String], [image_url:String], [num_frames:Integer], [frames_per_second:Integer], [seed:Integer], [motion:Integer], [resolution:ResolutionEnum], [num_inference_steps:Integer]",
        },
        "MiniMax (Hailuo AI) Video 01 Director - Image to Video Image to Video": {
            "endpoint": "fal-ai/minimax/video-01-director/image-to-video",
            "resolutions": [], "aspect_ratios": ["16:9", "9:16", "1:1"], "durations": [],
            "schema_str": "[Prompt:String], [Image_url:String], [prompt_optimizer:Boolean]",
        },
         "Skyreels V1 (Image-to-Video) Image to Video": {
            "endpoint": "fal-ai/skyreels-i2v",
            "resolutions": [], "aspect_ratios": ["16:9", "9:16", "1:1"], "durations": [],
            "schema_str": "[Prompt:String], [Image_url:String], [seed:Integer], [guidance_scale:Float], [num_inference_steps:Integer], [negative_prompt:String], [aspect_ratio:AspectRatioEnum]",
        },
        "Kling 1.6 Image to Video (Standard)": {
            "endpoint": "fal-ai/kling-video/v1.6/standard/image-to-video",
            "resolutions": [], "aspect_ratios": ["16:9", "9:16", "1:1"], "durations": [5, 10],
            "schema_str": "[Prompt:String], [Image_url:String], [duration:DurationEnum], [aspect_ratio:AspectRatioEnum]",
        },
         "MiniMax (Hailuo AI) Video 01 Live Image to Video": {
            "endpoint": "fal-ai/minimax/video-01-live/image-to-video",
            "resolutions": [], "aspect_ratios": [], "durations": [],
            "schema_str": "[Prompt:String], [Image_url:String], [prompt_optimizer:Boolean]",
         },
         "Kling 1.5 Image to Video (Pro)": {
            "endpoint": "fal-ai/kling-video/v1.5/pro/image-to-video",
            "resolutions": [], "aspect_ratios": ["16:9", "9:16", "1:1"], "durations": [5, 10],
            "schema_str": "[Prompt:String], [Image_url:String], [duration:DurationEnum], [aspect_ratio:AspectRatioEnum]",
        },
        "Pika Image to Video (v2.2) Image to Video": {
            "endpoint": "fal-ai/pika/v2.2/image-to-video",
            "resolutions": ["720p", "1080p"], "aspect_ratios": ["16:9", "9:16", "1:1", "4:5", "5:4", "3:2", "2:3"], "durations": [],
            "schema_str": "[Image_url:String], [Prompt:String], [seed:Integer], [negative_prompt:String], [resolution:ResolutionEnum], [duration:Integer]"
        },
        "Pika Image to Video (v2.1) Image to Video": {
            "endpoint": "fal-ai/pika/v2.1/image-to-video",
            "resolutions": ["720p", "1080p"], "aspect_ratios": ["16:9", "9:16", "1:1", "4:5", "5:4", "3:2", "2:3"], "durations": [],
            "schema_str": "[Image_url:String], [Prompt:String], [seed:Integer], [negative_prompt:String], [resolution:ResolutionEnum], [duration:Integer]"
        },
        "Vidu Image to Video": {
            "endpoint": "fal-ai/vidu/image-to-video",
            "resolutions": [], "aspect_ratios": ["16:9", "9:16", "1:1"], "durations": [3],
            "schema_str": "[image_url:String], [prompt:String], [negative_prompt:String], [seed:Integer], [num_frames:Integer], [fps:Integer], [guidance_scale:Float], [num_inference_steps:Integer]"
        },
         "WAN Pro Image to Video": {
            "endpoint": "fal-ai/wan-pro/image-to-video",
            "resolutions": ["480p", "720p"], "aspect_ratios": ["auto", "16:9", "9:16", "1:1"], "durations": [5],
            "schema_str": "[Prompt:String], [negative_prompt:String], [image_url:String], [num_frames:Integer], [frames_per_second:Integer], [seed:Integer], [motion:Integer], [resolution:ResolutionEnum], [num_inference_steps:Integer]"
        },
        "Hunyuan Video (Image to Video)": {
            "endpoint": "fal-ai/hunyuan-video-image-to-video",
            "resolutions": ["256p", "512p"], "aspect_ratios": ["1:1"], "durations": [],
            "schema_str": "[Prompt:String], [image_url:String], [seed:Integer], [negative_prompt:String]"
        },
        "LTX Video v0.95 Image to Video": {
            "endpoint": "fal-ai/ltx-video-v095/image-to-video",
            "resolutions": ["256p", "512p"], "aspect_ratios": ["16:9", "9:16", "1:1"], "durations": [2,3,4,5,6,7,8,9,10],
            "schema_str": "[Prompt:String], [image_url:String], [negative_prompt:String], [seed:Integer], [num_frames:Integer], [fps:Integer], [guidance_scale:Float], [num_inference_steps:Integer]"
        },
          "Luma Dream Machine (Image to Video) Image to Video": {
            "endpoint": "fal-ai/luma-dream-machine/image-to-video",
            "resolutions": ["540p", "720p", "1080p"], "aspect_ratios": ["16:9", "9:16", "4:3", "3:4", "21:9", "9:21"], "durations": [5],
            "schema_str": "[Prompt:String], [image_url:String], [end_image_url:String], [aspect_ratio:AspectRatioEnum], [loop:Boolean], [resolution:ResolutionEnum], [duration:DurationEnum]"
        },
        "Luma Ray 2 (Image to Video) Image to Video": {
            "endpoint": "fal-ai/luma-dream-machine/ray-2/image-to-video",
            "resolutions": ["540p", "720p", "1080p"], "aspect_ratios": ["16:9", "9:16", "4:3", "3:4", "21:9", "9:21"], "durations": [5],
            "schema_str": "[Prompt:String], [image_url:String], [end_image_url:String], [aspect_ratio:AspectRatioEnum], [loop:Boolean], [resolution:ResolutionEnum], [duration:DurationEnum]"
        },
        "Hunyuan Video (Image to Video - LoRA)": {
            "endpoint": "fal-ai/hunyuan-video-img2vid-lora",
            "resolutions": ["256p", "512p"], "aspect_ratios": ["1:1"], "durations": [],
            "schema_str": "[Prompt:String], [image_url:String], [seed:Integer], [negative_prompt:String]"
        },
        "PixVerse v3.5: Image to Video Image to Video": {
            "endpoint": "fal-ai/pixverse/v3.5/image-to-video",
            "resolutions": ["360p", "540p", "720p", "1080p"], "aspect_ratios": ["16:9", "4:3", "1:1", "3:4", "9:16"], "durations": [5, 8],
            "schema_str": "[Prompt:String], [aspect_ratio:AspectRatioEnum], [resolution:ResolutionEnum], [duration:DurationEnum], [negative_prompt:String], [style:Enum], [seed:Integer], [image_url:String]"
        },
        "PixVerse v3.5: Image to Video Fast Image to Video": {
            "endpoint": "fal-ai/pixverse/v3.5/image-to-video/fast",
            "resolutions": ["360p", "540p", "720p"], "aspect_ratios": ["16:9", "4:3", "1:1", "3:4", "9:16"], "durations": [],
            "schema_str": "[Prompt:String], [aspect_ratio:AspectRatioEnum], [resolution:ResolutionEnum], [negative_prompt:String], [style:Enum], [seed:Integer], [image_url:String]"
        },
         "LTX Video Image to Video": {
            "endpoint": "fal-ai/ltx-video/image-to-video",
            "resolutions": ["256p", "512p"], "aspect_ratios": ["16:9", "9:16", "1:1"], "durations": [2,3,4,5,6,7,8,9,10],
            "schema_str": "[Prompt:String], [image_url:String], [negative_prompt:String], [seed:Integer], [num_frames:Integer], [fps:Integer], [guidance_scale:Float], [num_inference_steps:Integer]"
        },
        "CogVideoX 5B Image to Video": {
            "endpoint": "fal-ai/cogvideox-5b/image-to-video",
            "resolutions": [], "aspect_ratios": ["16:9", "9:16", "1:1"], "durations": [2,3,4,5,6,7,8,9,10],
            "schema_str": "[Prompt:String], [image_url:String], [seed:Integer], [negative_prompt:String], [num_frames:Integer], [fps:Integer], [guidance_scale:Float], [num_inference_steps:Integer]"
        },
          "Kling 1 Image to Video (Pro)": {
            "endpoint": "fal-ai/kling-video/v1/pro/image-to-video",
            "resolutions": [], "aspect_ratios": ["16:9", "9:16", "1:1"], "durations": [5, 10],
            "schema_str": "[Prompt:String], [Image_url:String], [duration:DurationEnum], [aspect_ratio:AspectRatioEnum]",
        },
         "Kling 1 Image to Video (Standard)": {
            "endpoint": "fal-ai/kling-video/v1/standard/image-to-video",
            "resolutions": [], "aspect_ratios": ["16:9", "9:16", "1:1"], "durations": [5, 10],
            "schema_str": "[Prompt:String], [Image_url:String], [duration:DurationEnum], [aspect_ratio:AspectRatioEnum]",
        },
    },
    "text_to_video": {
        "Kling 1.6 Text to Video (Pro)": {
            "endpoint": "fal-ai/kling-video/v1.6/pro/text-to-video",
            "resolutions": [], "aspect_ratios": ["16:9", "9:16", "1:1"], "durations": [5, 10],
            "schema_str": "[Prompt:String], [negative_prompt:String], [duration:DurationEnum], [aspect_ratio:AspectRatioEnum], [seed:Integer], [width:Integer], [height:Integer], [motion_bucket_id:Integer], [cond_aug:Float], [steps:Integer], [guidance_scale:Float], [fps:Integer]",
        },
        "Pika Text to Video": {
            "endpoint": "fal-ai/pika/v2/text-to-video",
            "resolutions": ["720p", "1024p"], "aspect_ratios": ["16:9", "9:16", "1:1", "4:5", "5:4", "3:2", "2:3"], "durations": [],
            "schema_str": "[prompt:String], [negative_prompt:String], [seed:Integer], [resolution:ResolutionEnum], [duration:Integer]"
        },
        "Luma Dream Machine Text to Video": {
            "endpoint": "fal-ai/luma-dream-machine/text-to-video",
            "resolutions": ["540p", "720p", "1080p"], "aspect_ratios": ["16:9", "9:16", "4:3", "3:4", "21:9", "9:21"], "durations": [5],
            "schema_str": "[prompt:String], [seed:Integer], [aspect_ratio:AspectRatioEnum], [loop:Boolean], [resolution:ResolutionEnum], [duration:DurationEnum]"
        },
        "PixVerse v4 Text to Video": {
            "endpoint": "fal-ai/pixverse/v4/text-to-video",
            "resolutions": ["360p", "540p", "720p", "1080p"], "aspect_ratios": ["16:9", "4:3", "1:1", "3:4", "9:16"], "durations": [5, 8],
            "schema_str": "[prompt:String], [negative_prompt:String], [style:Enum], [seed:Integer], [aspect_ratio:AspectRatioEnum], [resolution:ResolutionEnum], [duration:DurationEnum]"
        },
        "MiniMax (Hailuo AI) Video 01 Text to Video": {
            "endpoint": "fal-ai/minimax/video-01/text-to-video",
            "resolutions": [], "aspect_ratios": [], "durations": [],
            "schema_str": "[prompt:String]"
        },
        "Hunyuan Video Text to Video": {
            "endpoint": "fal-ai/hunyuan-video",
            "resolutions": ["256p", "512p"], "aspect_ratios": ["1:1"], "durations": [],
            "schema_str": "[prompt:String], [seed:Integer], [negative_prompt:String]"
        },
    },
}

# --- Polling Helper Function ---
def _poll_fal_job(endpoint_id, request_id, polling_interval=3, timeout=900): # Default timeout 15 mins
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
            # Attempt cancellation before raising timeout
            print(f"[Fal Poller] Attempting to cancel timed out job {request_id}...")
            try:
                fal_client.cancel(endpoint_id, request_id)
                print(f"[Fal Poller] Cancel request sent for timed out job {request_id}.")
            except Exception as cancel_e:
                print(f"WARN: [Fal Poller] Failed to send cancel request after timeout: {cancel_e}")
            raise TimeoutError(f"Fal.ai job {request_id} timed out after {timeout}s")

        # --- Status Check ---
        try:
            status_response = fal_client.status(endpoint_id, request_id, logs=False)
            status = status_response.get('status')
            queue_pos = status_response.get('queue_position')

            print(f"[Fal Poller] Job {request_id}: Status={status}, Queue={queue_pos if queue_pos is not None else 'N/A'}, Elapsed={elapsed_time:.1f}s")

            if status == "COMPLETED":
                print(f"[Fal Poller] Job {request_id} completed.")
                final_result = fal_client.result(endpoint_id, request_id)
                return final_result

            elif status in ["ERROR", "FAILED", "CANCELLED"]:
                error_message = f"Fal.ai job {request_id} failed with status: {status}"
                print(f"ERROR: [Fal Poller] {error_message}")
                raise RuntimeError(error_message)

            elif status in ["IN_QUEUE", "IN_PROGRESS"]:
                try:
                    time.sleep(polling_interval)
                except KeyboardInterrupt:
                    print(f"\nWARN: [Fal Poller] KeyboardInterrupt caught during sleep for job {request_id}. Attempting cancellation...")
                    raise KeyboardInterrupt # Re-raise

            else: # Unknown status
                print(f"WARN: [Fal Poller] Job {request_id} has unknown status: {status}. Continuing poll.")
                try:
                    time.sleep(polling_interval)
                except KeyboardInterrupt:
                     print(f"\nWARN: [Fal Poller] KeyboardInterrupt during sleep (unknown status) for job {request_id}. Attempting cancellation...")
                     raise KeyboardInterrupt # Re-raise

        except KeyboardInterrupt: # Catch during API call itself
            print(f"\nWARN: [Fal Poller] KeyboardInterrupt caught during status check for job {request_id}. Attempting cancellation...")
            raise KeyboardInterrupt # Re-raise
        except Exception as e:
            # Handle potential network errors during status check, but keep polling unless it's fatal
            if isinstance(e, requests.exceptions.RequestException):
                print(f"WARN: [Fal Poller] Network error checking status for job {request_id}: {e}. Retrying...")
                try:
                    time.sleep(polling_interval * 2) # Longer sleep after network error
                except KeyboardInterrupt:
                    print(f"\nWARN: [Fal Poller] KeyboardInterrupt during network error backoff for job {request_id}. Attempting cancellation...")
                    raise KeyboardInterrupt
            else:
                # For other unexpected errors during status check, raise them
                print(f"ERROR: [Fal Poller] Unexpected error polling job {request_id}: {e}")
                traceback.print_exc()
                raise e

# --- Helper Function to Parse Schema String ---
def parse_schema(schema_str):
    params = set()
    parts = schema_str.strip('[]').split('], [')
    for part in parts:
        if ':' in part:
            param_name = part.split(':')[0].strip()
            if param_name == "Image_url": param_name = "image_url"
            if param_name == "Prompt": param_name = "prompt"
            if param_name == "duration": param_name = "duration_seconds"
            if param_name == "num_inference_steps": param_name = "steps"
            if param_name == "resolution": param_name = "resolution_enum"
            if param_name == "aspect_ratio": param_name = "aspect_ratio_enum"
            params.add(param_name)
    return params

# --- Populate Model Configs with Parsed Schema ---
for category, models in MODEL_CONFIGS.items():
    for name, config in models.items():
        config['expected_params'] = parse_schema(config['schema_str'])

# --- Dynamically Create Dropdown Lists ---
ALL_MODEL_NAMES_I2V = sorted(list(MODEL_CONFIGS["image_to_video"].keys()))
ALL_MODEL_NAMES_T2V = sorted(list(MODEL_CONFIGS["text_to_video"].keys()))
ALL_RESOLUTIONS = sorted(list(set(res for cat in MODEL_CONFIGS.values() for cfg in cat.values() for res in cfg['resolutions'] if res))) # Ensure no empty strings
ALL_ASPECT_RATIOS = sorted(list(set(ar for cat in MODEL_CONFIGS.values() for cfg in cat.values() for ar in cfg['aspect_ratios'] if ar))) # Ensure no empty strings
if not ALL_RESOLUTIONS: ALL_RESOLUTIONS = ["720p", "1080p", "512p", "576p"]
if not ALL_ASPECT_RATIOS: ALL_ASPECT_RATIOS = ["16:9", "9:16", "1:1", "4:3", "3:4"]
if "auto" not in ALL_ASPECT_RATIOS: ALL_ASPECT_RATIOS.insert(0, "auto")
ALL_RESOLUTIONS.insert(0, "auto")


# --- Helper Functions with Corrected Logging ---
def _prepare_image_bytes(image_tensor):
    """Converts ComfyUI Image Tensor to PNG bytes."""
    if image_tensor is None: print("[Fal Helper] No image tensor provided."); return None, None
    print("[Fal Helper] Preparing image tensor...")
    try:
        if image_tensor.dim() == 4 and image_tensor.shape[0] == 1: img_tensor = image_tensor.squeeze(0)
        elif image_tensor.dim() == 3: img_tensor = image_tensor
        else: raise ValueError(f"Unexpected image tensor shape: {image_tensor.shape}")
        img_tensor = img_tensor.cpu(); img_np = img_tensor.numpy()
        if img_np.max() <= 1.0 and img_np.min() >= 0.0: img_np = (img_np * 255)
        img_np = img_np.astype(np.uint8); pil_image = Image.fromarray(img_np, 'RGB')
        buffered = io.BytesIO(); pil_image.save(buffered, format="PNG"); img_bytes = buffered.getvalue()
        print(f"[Fal Helper] Image tensor prep complete ({len(img_bytes)} bytes).")
        return img_bytes, "image/png"
    except Exception as e: print(f"ERROR: [Fal Helper] Image tensor processing failed: {e}"); traceback.print_exc(); return None, None

def _save_tensor_to_temp_video(image_tensor_batch, fps=30):
    """Saves a ComfyUI Image Tensor Batch (B, H, W, C) to a temporary MP4 file."""
    if image_tensor_batch is None or image_tensor_batch.dim() != 4 or image_tensor_batch.shape[0] == 0: print("[Fal Helper] Invalid video tensor batch."); return None
    print("[Fal Helper] Saving video tensor batch to temp file...")
    batch_size, height, width, channels = image_tensor_batch.shape
    if channels != 3: print(f"ERROR: [Fal Helper] Expected 3 channels, got {channels}."); return None
    output_dir = folder_paths.get_temp_directory(); os.makedirs(output_dir, exist_ok=True)
    filename = f"fal_temp_upload_vid_{uuid.uuid4().hex}.mp4"
    temp_video_filepath = os.path.join(output_dir, filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v'); video_writer = cv2.VideoWriter(temp_video_filepath, fourcc, float(fps), (width, height))
    if not video_writer.isOpened(): print(f"ERROR: [Fal Helper] Failed to open video writer: {temp_video_filepath}"); return None
    try:
        image_tensor_batch_cpu = image_tensor_batch.cpu()
        for i in range(batch_size):
            frame_tensor = image_tensor_batch_cpu[i]; frame_np = frame_tensor.numpy()
            if frame_np.max() <= 1.0 and frame_np.min() >= 0.0: frame_np = (frame_np * 255)
            frame_np = frame_np.astype(np.uint8); frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
        print(f"[Fal Helper] Temp video saved: {temp_video_filepath}")
        return temp_video_filepath
    except Exception as e: print(f"ERROR: [Fal Helper] Video writing failed: {e}"); traceback.print_exc()
    finally:
        if video_writer.isOpened(): video_writer.release()
        # Cleanup partial file only if error occurred during writing
        if 'e' in locals() and os.path.exists(temp_video_filepath): # Check if exception 'e' exists
             try: os.remove(temp_video_filepath); print(f"[Fal Helper] Cleaned partial temp video: {temp_video_filepath}")
             except Exception as clean_e: print(f"WARN: [Fal Helper] Cleanup failed on video write error: {clean_e}")
    return None # Return None if exception occurred

def _upload_media_to_fal(media_bytes, filename_hint, content_type):
    """Saves media bytes to temp file, uploads via fal_client.upload_file."""
    if not media_bytes: print(f"ERROR: [Fal Helper] No media bytes for upload ({filename_hint})."); return None
    temp_path = None
    try:
        temp_dir = folder_paths.get_temp_directory(); os.makedirs(temp_dir, exist_ok=True)
        ext = os.path.splitext(filename_hint)[1]
        if not ext and content_type:
             ct_map = {'png':'.png', 'jpeg':'.jpg', 'jpg':'.jpg', 'mp4':'.mp4', 'webm':'.webm', 'mp3':'.mp3', 'mpeg':'.mp3', 'wav':'.wav'}
             for key, val in ct_map.items():
                 if key in content_type: ext = val; break
        if not ext: ext = ".tmp"
        temp_filename = f"fal_upload_{uuid.uuid4().hex}{ext}"; temp_path = os.path.join(temp_dir, temp_filename)
        print(f"[Fal Helper] Writing temp file: {temp_path} ({len(media_bytes)} bytes)")
        with open(temp_path, "wb") as f: f.write(media_bytes)
        print(f"[Fal Helper] Uploading {temp_path}...")
        file_url = fal_client.upload_file(temp_path)
        print(f"[Fal Helper] Upload successful: {file_url}")
        return file_url
    except Exception as e: print(f"ERROR: [Fal Helper] Upload failed for {filename_hint}: {e}"); traceback.print_exc(); return None
    finally:
        if temp_path and os.path.exists(temp_path):
            try: print(f"[Fal Helper] Cleaning temp upload: {temp_path}"); os.remove(temp_path)
            except Exception as cleanup_e: print(f"WARN: [Fal Helper] Temp upload cleanup failed: {cleanup_e}")

def _save_audio_tensor_to_temp_wav(audio_data):
    """Saves ComfyUI AUDIO dict to temp WAV."""
    # print(f"[Fal Helper] _save_audio... received keys: {audio_data.keys() if isinstance(audio_data, dict) else 'Not dict'}") # Debug
    if not isinstance(audio_data, dict) or 'sample_rate' not in audio_data or \
       ('samples' not in audio_data and 'waveform' not in audio_data):
        print("ERROR: [Fal Helper] Invalid audio data format."); return None
    sample_rate = audio_data['sample_rate']
    samples_tensor = audio_data.get('samples') or audio_data.get('waveform')
    if samples_tensor is None: print("ERROR: [Fal Helper] No audio tensor found."); return None
    print(f"[Fal Helper] Processing audio tensor (Rate: {sample_rate}, Shape: {samples_tensor.shape})")
    try:
        samples_tensor = samples_tensor.cpu()
        if samples_tensor.dim() == 3: print("[Fal Helper] Using first audio item from batch."); samples_tensor = samples_tensor[0]
        elif samples_tensor.dim() != 2: raise ValueError("Unexpected audio tensor dimensions.")
        samples_np = samples_tensor.numpy().T
        if np.issubdtype(samples_np.dtype, np.floating):
             print("[Fal Helper] Converting float audio to int16."); samples_np = np.clip(samples_np, -1.0, 1.0); samples_np = (samples_np * 32767).astype(np.int16)
        elif not np.issubdtype(samples_np.dtype, np.integer): print(f"WARN: [Fal Helper] Unknown audio dtype: {samples_np.dtype}"); samples_np = samples_np.astype(np.int16)
        output_dir = folder_paths.get_temp_directory(); os.makedirs(output_dir, exist_ok=True)
        filename = f"fal_temp_upload_aud_{uuid.uuid4().hex}.wav"
        temp_audio_filepath = os.path.join(output_dir, filename)
        print(f"[Fal Helper] Saving temp WAV: {temp_audio_filepath}")
        scipy.io.wavfile.write(temp_audio_filepath, sample_rate, samples_np)
        return temp_audio_filepath
    except Exception as e: print(f"ERROR: [Fal Helper] Failed saving audio tensor: {e}"); traceback.print_exc(); return None


# --- Define the Image-to-Video Node Class with Polling ---
class FalAPIVideoGeneratorI2V:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { "model_name": (ALL_MODEL_NAMES_I2V,), "api_key": ("STRING", {"multiline": False, "default": "Paste FAL_KEY credentials here (e.g., key_id:key_secret)"}), "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}), "prompt": ("STRING", {"multiline": True, "default": "A wild Burgstall appears"}), },
            "optional": { "image": ("IMAGE",), "negative_prompt": ("STRING", {"multiline": True, "default": "Ugly, blurred, distorted"}), "resolution_enum": (ALL_RESOLUTIONS, {"default": "auto"}), "aspect_ratio_enum": (ALL_ASPECT_RATIOS, {"default": "auto"}), "duration_seconds": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 30.0, "step": 0.5}), "guidance_scale": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 20.0, "step": 0.1}), "steps": ("INT", {"default": 25, "min": 1, "max": 100, "step": 1}), "num_frames": ("INT", {"default": 0, "min": 0, "max": 1000}), "style": (["auto", "cinematic", "anime", "photorealistic", "fantasy", "cartoon"], {"default": "auto"}), "prompt_optimizer": ("BOOLEAN", {"default": False}), "cleanup_temp_video": ("BOOLEAN", {"default": True}), }
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_batch",)
    FUNCTION = "generate_video"
    CATEGORY = "BS_FalAi-API-Video/Image-to-Video"

    # Specific Base64 image prep for this node type
    def _prepare_image(self, image_tensor, target_width=None, target_height=None):
        if image_tensor is None: print("FalAPIVideoGeneratorI2V: No image provided."); return None
        print("FalAPIVideoGeneratorI2V: Preparing image (Base64 method)...")
        try:
            if image_tensor.dim() == 4 and image_tensor.shape[0] == 1: img_tensor = image_tensor[0]
            elif image_tensor.dim() == 3: img_tensor = image_tensor
            else: raise ValueError(f"Unexpected shape: {image_tensor.shape}")
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8); pil_image = Image.fromarray(img_np)
            if target_width and target_height and (pil_image.width != target_width or pil_image.height != target_height):
                pil_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
            buffered = io.BytesIO(); pil_image.save(buffered, format="PNG"); img_bytes = buffered.getvalue()
            img_data_uri = f"data:image/png;base64,{base64.b64encode(img_bytes).decode('utf-8')}"
            print("FalAPIVideoGeneratorI2V: Image prep complete (Base64).")
            return img_data_uri
        except Exception as e: print(f"ERROR: FalAPIVideoGeneratorI2V: Image processing failed: {e}"); traceback.print_exc(); return None

    def generate_video(self, model_name, api_key, seed, prompt, image=None, negative_prompt=None, resolution_enum="auto", aspect_ratio_enum="auto", duration_seconds=5.0, guidance_scale=7.5, steps=25, num_frames=0, style="auto", prompt_optimizer=False, cleanup_temp_video=True):
        def log_prefix(): return "FalAPIVideoGeneratorI2V:"
        print(f"{log_prefix()} Starting generation...")

        # --- 1. Config & API Key ---
        if model_name not in MODEL_CONFIGS["image_to_video"]: print(f"ERROR: {log_prefix()} Unknown model '{model_name}'"); return (None,)
        config = MODEL_CONFIGS["image_to_video"][model_name]; endpoint_id = config['endpoint']; expected_params = config['expected_params']
        print(f"{log_prefix()} Model: {model_name}, Endpoint: {endpoint_id}")
        if not api_key or not api_key.strip() or api_key == "Paste FAL_KEY credentials here (e.g., key_id:key_secret)": print(f"ERROR: {log_prefix()} API Key missing."); return (None,)
        api_key_value = api_key.strip()
        try: os.environ["FAL_KEY"] = api_key_value; print(f"{log_prefix()} Using API Key.")
        except Exception as e: print(f"ERROR: {log_prefix()} Failed setting API Key: {e}"); traceback.print_exc(); return (None,)

        # --- 2. Payload ---
        payload = {}
        if 'image_url' in expected_params:
            if image is not None:
                img_data_uri = self._prepare_image(image)
                if img_data_uri: payload['image_url'] = img_data_uri
                else: print(f"ERROR: {log_prefix()} Failed preparing image."); return (None,)
            else: print(f"WARN: {log_prefix()} Model expects 'image_url', none provided.")
        elif image is not None: print(f"WARN: {log_prefix()} Image provided but model doesn't expect 'image_url'.")

        if 'prompt' in expected_params: payload['prompt'] = prompt.strip() if prompt else ""
        if 'negative_prompt' in expected_params and negative_prompt: payload['negative_prompt'] = negative_prompt.strip()

        param_map = { "seed": seed, "duration_seconds": duration_seconds, "guidance_scale": guidance_scale, "steps": steps, "num_frames": num_frames, "style": style, "prompt_optimizer": prompt_optimizer, "resolution_enum": resolution_enum, "aspect_ratio_enum": aspect_ratio_enum }
        api_name_map = { "duration_seconds": "duration", "steps": "num_inference_steps", "resolution_enum": "resolution", "aspect_ratio_enum": "aspect_ratio" }
        for input_name, value in param_map.items():
            api_name = api_name_map.get(input_name, input_name)
            if api_name in expected_params:
                if isinstance(value, str) and value.lower() == "auto": continue
                if isinstance(value, (int, float)) and value == 0 and input_name in ["num_frames", "seed"] and not (input_name == "seed" and 0 in expected_params): continue
                try: # Add specific type conversions here if needed based on future schema inspection
                    if api_name == "duration" and "duration:Integer" in config.get('schema_str', ''): payload[api_name] = int(value)
                    elif api_name == "steps" and "steps:Integer" in config.get('schema_str', ''): payload[api_name] = int(value) # Assuming steps maps to num_inference_steps of type Integer
                    else: payload[api_name] = value
                except (ValueError, TypeError) as te: print(f"WARN: {log_prefix()} Type conversion failed for {api_name}={value}: {te}")

        if 'resolution' in expected_params and resolution_enum != "auto": payload['resolution'] = resolution_enum; payload.pop('width', None); payload.pop('height', None)
        if 'aspect_ratio' in expected_params and aspect_ratio_enum != "auto": payload['aspect_ratio'] = aspect_ratio_enum; payload.pop('width', None); payload.pop('height', None)

        # --- 3. API Call & Processing ---
        request_id = None; video_url = None; temp_video_filepath = None; frames_tensor = None
        try:
            print(f"{log_prefix()} Submitting job. Payload keys: {list(payload.keys())}")
            if not endpoint_id: raise ValueError("Endpoint ID missing")
            handler = fal_client.submit(endpoint_id.strip(), arguments=payload); request_id = handler.request_id
            print(f"{log_prefix()} Job submitted. Request ID: {request_id}")
            response = _poll_fal_job(endpoint_id, request_id, timeout=900) # Poll
            print(f"{log_prefix()} Job {request_id} completed.")

            video_data = response.get('video'); url_found = False
            if isinstance(video_data, dict): video_url = video_data.get('url')
            if not video_url and isinstance(response.get('videos'), list) and len(response['videos']) > 0 and isinstance(response['videos'][0], dict): video_url = response['videos'][0].get('url')
            if not video_url and isinstance(response, dict) and response.get('url') and isinstance(response.get('url'), str) and any(ext in response['url'].lower() for ext in ['.mp4', '.webm']): video_url = response['url']
            if not video_url: raise ValueError(f"No video URL found in result: {json.dumps(response, indent=2)}")
            print(f"{log_prefix()} Video URL: {video_url}")

            print(f"{log_prefix()} Downloading video...")
            video_response = requests.get(video_url, stream=True, timeout=300); video_response.raise_for_status()
            output_dir = folder_paths.get_temp_directory(); os.makedirs(output_dir, exist_ok=True)
            content_type = video_response.headers.get('content-type', '').lower()
            extension = '.mp4';
            if 'webm' in content_type or video_url.lower().endswith('.webm'): extension = '.webm'
            filename = f"fal_api_i2v_temp_{uuid.uuid4().hex}{extension}"
            temp_video_filepath = os.path.join(output_dir, filename)
            with open(temp_video_filepath, 'wb') as f:
                for chunk in video_response.iter_content(chunk_size=1024*1024): f.write(chunk)
            print(f"{log_prefix()} Video downloaded: {temp_video_filepath}")

            print(f"{log_prefix()} Extracting frames...")
            frames_list = []; cap = cv2.VideoCapture(temp_video_filepath)
            if not cap.isOpened(): raise IOError(f"Cannot open video: {temp_video_filepath}")
            while True:
                ret, frame = cap.read()
                if not ret: break
                frames_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            if not frames_list: raise ValueError(f"No frames extracted: {temp_video_filepath}")
            print(f"{log_prefix()} Extracted {len(frames_list)} frames.")
            frames_np = np.stack(frames_list, axis=0); frames_tensor = torch.from_numpy(frames_np).float() / 255.0

            if frames_tensor is None: raise ValueError("Frame tensor creation failed.")
            print(f"{log_prefix()} Completed. Tensor shape: {frames_tensor.shape}")
            return (frames_tensor,)

      
        except KeyboardInterrupt:
            print(f"ERROR: {log_prefix()} Execution interrupted by user.")
            if request_id:
                print(f"{log_prefix()} Attempting to cancel Fal.ai job {request_id}...")
                try:
                    fal_client.cancel(endpoint_id, request_id)
                    print(f"{log_prefix()} Fal.ai cancel request sent for job {request_id}.")
                except Exception as cancel_e:
                    print(f"WARN: {log_prefix()} Failed to send cancel request: {cancel_e}")
            return (None,) # Use (None,) for I2V/T2V/Omni
        
        except TimeoutError as e:
            print(f"ERROR: {log_prefix()} Job timed out: {e}")
            if request_id:
                print(f"{log_prefix()} Attempting to cancel Fal.ai job {request_id} due to timeout...")
                try:
                    fal_client.cancel(endpoint_id, request_id)
                    print(f"{log_prefix()} Fal.ai cancel request sent for job {request_id}.")
                except Exception as cancel_e:
                    print(f"WARN: {log_prefix()} Failed to send cancel request after timeout: {cancel_e}")
            return (None,) # Use (None,) for I2V/T2V/Omni
        
            
        except RuntimeError as e: print(f"ERROR: {log_prefix()} Fal.ai job failed: {e}"); return (None,)
        except requests.exceptions.RequestException as e: print(f"ERROR: {log_prefix()} Network error: {e}"); traceback.print_exc(); return (None,)
        except (cv2.error, IOError, ValueError, Image.UnidentifiedImageError) as e: print(f"ERROR: {log_prefix()} Media processing error: {e}"); traceback.print_exc(); return (None,)
        except Exception as e: req_id_str=f"Req ID: {request_id}" if request_id else 'N/A'; print(f"ERROR: {log_prefix()} Unexpected error ({req_id_str}): {e}"); traceback.print_exc(); return (None,)
        finally:
            if cleanup_temp_video and temp_video_filepath and os.path.exists(temp_video_filepath):
                try: print(f"{log_prefix()} Cleaning temp: {temp_video_filepath}"); os.remove(temp_video_filepath)
                except Exception as e: print(f"WARN: {log_prefix()} Temp cleanup failed: {e}")
            elif temp_video_filepath and os.path.exists(temp_video_filepath): print(f"{log_prefix()} Keeping temp: {temp_video_filepath}")


# --- Define the Text-to-Video Node Class with Polling ---
class FalAPIVideoGeneratorT2V:
    @classmethod
    def INPUT_TYPES(cls):
        # Keep original INPUT_TYPES definition
         return {
            "required": { "model_name": (ALL_MODEL_NAMES_T2V,), "api_key": ("STRING", {"multiline": False, "default": "Paste FAL_KEY credentials here (e.g., key_id:key_secret)"}), "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}), "prompt": ("STRING", {"multiline": True, "default": "A wild Burgstall appears"}), },
            "optional": { "negative_prompt": ("STRING", {"multiline": True, "default": "Blurry, low quality, text, watermark"}), "resolution_enum": (ALL_RESOLUTIONS, {"default": "auto"}), "aspect_ratio_enum": (ALL_ASPECT_RATIOS, {"default": "auto"}), "duration_seconds": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 30.0, "step": 0.5}), "guidance_scale": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 20.0, "step": 0.1}), "steps": ("INT", {"default": 25, "min": 1, "max": 100, "step": 1}), "num_frames": ("INT", {"default": 0, "min": 0, "max": 1000}), "style": (["auto", "cinematic", "anime", "photorealistic", "fantasy", "cartoon"], {"default": "auto"}), "prompt_optimizer": ("BOOLEAN", {"default": False}), "cleanup_temp_video": ("BOOLEAN", {"default": True}), }
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_batch",)
    FUNCTION = "generate_video"
    CATEGORY = "BS_FalAi-API-Video/Text-to-Video"

    def generate_video(self, model_name, api_key, seed, prompt, negative_prompt=None, resolution_enum="auto", aspect_ratio_enum="auto", duration_seconds=5.0, guidance_scale=7.5, steps=25, num_frames=0, style="auto", prompt_optimizer=False, cleanup_temp_video=True):
        def log_prefix(): return "FalAPIVideoGeneratorT2V:"
        print(f"{log_prefix()} Starting generation...")

        # --- 1. Config & API Key ---
        if model_name not in MODEL_CONFIGS["text_to_video"]: print(f"ERROR: {log_prefix()} Unknown model '{model_name}'"); return (None,)
        config = MODEL_CONFIGS["text_to_video"][model_name]; endpoint_id = config['endpoint']; expected_params = config['expected_params']
        print(f"{log_prefix()} Model: {model_name}, Endpoint: {endpoint_id}")
        if not api_key or not api_key.strip() or api_key == "Paste FAL_KEY credentials here (e.g., key_id:key_secret)": print(f"ERROR: {log_prefix()} API Key missing."); return (None,)
        api_key_value = api_key.strip()
        try: os.environ["FAL_KEY"] = api_key_value; print(f"{log_prefix()} Using API Key.")
        except Exception as e: print(f"ERROR: {log_prefix()} Failed setting API Key: {e}"); traceback.print_exc(); return (None,)

        # --- 2. Payload ---
        payload = {}
        if 'prompt' in expected_params: payload['prompt'] = prompt.strip() if prompt else ""
        if 'negative_prompt' in expected_params and negative_prompt: payload['negative_prompt'] = negative_prompt.strip()
        param_map = { "seed": seed, "duration_seconds": duration_seconds, "guidance_scale": guidance_scale, "steps": steps, "num_frames": num_frames, "style": style, "prompt_optimizer": prompt_optimizer, "resolution_enum": resolution_enum, "aspect_ratio_enum": aspect_ratio_enum }
        api_name_map = { "duration_seconds": "duration", "steps": "num_inference_steps", "resolution_enum": "resolution", "aspect_ratio_enum": "aspect_ratio" }
        for input_name, value in param_map.items():
            api_name = api_name_map.get(input_name, input_name)
            if api_name in expected_params:
                if isinstance(value, str) and value.lower() == "auto": continue
                if isinstance(value, (int, float)) and value == 0 and input_name in ["num_frames", "seed"] and not (input_name == "seed" and 0 in expected_params): continue
                try:
                    if api_name == "duration" and "duration:Integer" in config.get('schema_str', ''): payload[api_name] = int(value)
                    elif api_name == "steps" and "steps:Integer" in config.get('schema_str', ''): payload[api_name] = int(value)
                    else: payload[api_name] = value
                except (ValueError, TypeError) as te: print(f"WARN: {log_prefix()} Type conversion failed for {api_name}={value}: {te}")
        if 'resolution' in expected_params and resolution_enum != "auto": payload['resolution'] = resolution_enum; payload.pop('width', None); payload.pop('height', None)
        if 'aspect_ratio' in expected_params and aspect_ratio_enum != "auto": payload['aspect_ratio'] = aspect_ratio_enum; payload.pop('width', None); payload.pop('height', None)

        # --- 3. API Call & Processing ---
        request_id = None; video_url = None; temp_video_filepath = None; frames_tensor = None
        try:
            print(f"{log_prefix()} Submitting job. Payload keys: {list(payload.keys())}")
            if not endpoint_id: raise ValueError("Endpoint ID missing")
            handler = fal_client.submit(endpoint_id.strip(), arguments=payload); request_id = handler.request_id
            print(f"{log_prefix()} Job submitted. Request ID: {request_id}")
            response = _poll_fal_job(endpoint_id, request_id, timeout=900) # Poll
            print(f"{log_prefix()} Job {request_id} completed.")

            video_data = response.get('video'); url_found = False
            if isinstance(video_data, dict): video_url = video_data.get('url')
            if not video_url and isinstance(response.get('videos'), list) and len(response['videos']) > 0 and isinstance(response['videos'][0], dict): video_url = response['videos'][0].get('url')
            if not video_url and isinstance(response, dict) and response.get('url') and isinstance(response.get('url'), str) and any(ext in response['url'].lower() for ext in ['.mp4', '.webm']): video_url = response['url']
            if not video_url: raise ValueError(f"No video URL found in result: {json.dumps(response, indent=2)}")
            print(f"{log_prefix()} Video URL: {video_url}")

            print(f"{log_prefix()} Downloading video...")
            video_response = requests.get(video_url, stream=True, timeout=300); video_response.raise_for_status()
            output_dir = folder_paths.get_temp_directory(); os.makedirs(output_dir, exist_ok=True)
            content_type = video_response.headers.get('content-type', '').lower()
            extension = '.mp4';
            if 'webm' in content_type or video_url.lower().endswith('.webm'): extension = '.webm'
            filename = f"fal_api_t2v_temp_{uuid.uuid4().hex}{extension}"
            temp_video_filepath = os.path.join(output_dir, filename)
            with open(temp_video_filepath, 'wb') as f:
                for chunk in video_response.iter_content(chunk_size=1024*1024): f.write(chunk)
            print(f"{log_prefix()} Video downloaded: {temp_video_filepath}")

            print(f"{log_prefix()} Extracting frames...")
            frames_list = []; cap = cv2.VideoCapture(temp_video_filepath)
            if not cap.isOpened(): raise IOError(f"Cannot open video: {temp_video_filepath}")
            while True:
                ret, frame = cap.read()
                if not ret: break
                frames_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            if not frames_list: raise ValueError(f"No frames extracted: {temp_video_filepath}")
            print(f"{log_prefix()} Extracted {len(frames_list)} frames.")
            frames_np = np.stack(frames_list, axis=0); frames_tensor = torch.from_numpy(frames_np).float() / 255.0

            if frames_tensor is None: raise ValueError("Frame tensor creation failed.")
            print(f"{log_prefix()} Completed. Tensor shape: {frames_tensor.shape}")
            return (frames_tensor,)

      
        except KeyboardInterrupt:
            print(f"ERROR: {log_prefix()} Execution interrupted by user.")
            if request_id:
                print(f"{log_prefix()} Attempting to cancel Fal.ai job {request_id}...")
                try:
                    fal_client.cancel(endpoint_id, request_id)
                    print(f"{log_prefix()} Fal.ai cancel request sent for job {request_id}.")
                except Exception as cancel_e:
                    print(f"WARN: {log_prefix()} Failed to send cancel request: {cancel_e}")
            return (None,) # Use (None,) for I2V/T2V/Omni
        
        except TimeoutError as e:
            print(f"ERROR: {log_prefix()} Job timed out: {e}")
            if request_id:
                print(f"{log_prefix()} Attempting to cancel Fal.ai job {request_id} due to timeout...")
                try:
                    fal_client.cancel(endpoint_id, request_id)
                    print(f"{log_prefix()} Fal.ai cancel request sent for job {request_id}.")
                except Exception as cancel_e:
                    print(f"WARN: {log_prefix()} Failed to send cancel request after timeout: {cancel_e}")
            return (None,) # Use (None,) for I2V/T2V/Omni
        
            
        except RuntimeError as e: print(f"ERROR: {log_prefix()} Fal.ai job failed: {e}"); return (None,)
        except requests.exceptions.RequestException as e: print(f"ERROR: {log_prefix()} Network error: {e}"); traceback.print_exc(); return (None,)
        except (cv2.error, IOError, ValueError, Image.UnidentifiedImageError) as e: print(f"ERROR: {log_prefix()} Media processing error: {e}"); traceback.print_exc(); return (None,)
        except Exception as e: req_id_str=f"Req ID: {request_id}" if request_id else 'N/A'; print(f"ERROR: {log_prefix()} Unexpected error ({req_id_str}): {e}"); traceback.print_exc(); return (None,)
        finally:
            if cleanup_temp_video and temp_video_filepath and os.path.exists(temp_video_filepath):
                try: print(f"{log_prefix()} Cleaning temp: {temp_video_filepath}"); os.remove(temp_video_filepath)
                except Exception as e: print(f"WARN: {log_prefix()} Temp cleanup failed: {e}")
            elif temp_video_filepath and os.path.exists(temp_video_filepath): print(f"{log_prefix()} Keeping temp: {temp_video_filepath}")


# --- Define the Omni Pro Node Class with Polling ---
class FalAPIOmniProNode:
    AUTO_KEY_START_IMAGE = "image_url"
    AUTO_KEY_END_IMAGE = "end_image_url"
    AUTO_KEY_INPUT_VIDEO = "video_url"
    AUTO_KEY_INPUT_AUDIO = "audio_url"

    @classmethod
    def INPUT_TYPES(cls):
        # Keep original INPUT_TYPES definition
        return {
            "required": { "endpoint_id": ("STRING", {"multiline": False, "default": "fal-ai/some-model/endpoint-id"}), "api_key": ("STRING", {"multiline": False, "default": "Paste FAL_KEY credentials here (e.g., key_id:key_secret)"}), "parameters_json": ("STRING", {"multiline": True, "default": json.dumps({"prompt": "A description", "seed": 12345}, indent=2)}), },
            "optional": { "start_image": ("IMAGE",), "end_image": ("IMAGE",), "input_video": ("IMAGE",), "input_audio": ("AUDIO",), "cleanup_temp_files": ("BOOLEAN", {"default": True}), "output_video_fps": ("INT", {"default": 30, "min": 1, "max": 120}), }
        }
    RETURN_TYPES = ("IMAGE",) # Only returns image batch
    RETURN_NAMES = ("image_batch",)
    FUNCTION = "execute_omni_request"
    CATEGORY = "BS_FalAi-API-Omni"

    def execute_omni_request(self, endpoint_id, api_key, parameters_json, start_image=None, end_image=None, input_video=None, input_audio=None, cleanup_temp_files=True, output_video_fps=30):
        def log_prefix(): return "FalAPIOmniProNode:"
        print(f"{log_prefix()} Starting request...")

        # --- Setup & Initializations ---
        uploaded_media_urls = {}; temp_files_to_clean = []; final_payload = {}
        temp_download_filepath = None; frames_tensor = None; img_tensor = None
        request_id = None

        # --- 1. API Key & Params ---
        if not api_key or not api_key.strip() or api_key == "Paste FAL_KEY credentials here (e.g., key_id:key_secret)": print(f"ERROR: {log_prefix()} API Key missing."); return (None,)
        api_key_value = api_key.strip()
        try: os.environ["FAL_KEY"] = api_key_value; print(f"{log_prefix()} Using API Key.")
        except Exception as e: print(f"ERROR: {log_prefix()} Failed setting API Key: {e}"); traceback.print_exc(); return (None,)
        user_params = {}
        try:
            if parameters_json and parameters_json.strip(): user_params = json.loads(parameters_json)
            if not isinstance(user_params, dict): raise ValueError("JSON must be a dict")
            print(f"{log_prefix()} Parsed parameters JSON.")
        except Exception as e: print(f"ERROR: {log_prefix()} Invalid JSON: {e}"); return (None,)

        # --- 2. Media Uploads ---
        upload_error = False
        try:
            if start_image is not None:
                img_bytes, ct = _prepare_image_bytes(start_image)
                if img_bytes: url = _upload_media_to_fal(img_bytes, "start_img.png", ct); uploaded_media_urls[self.AUTO_KEY_START_IMAGE] = url;
                if not img_bytes or not url: upload_error = True; print(f"ERROR: {log_prefix()} Start image prep/upload failed.")
            if end_image is not None and not upload_error:
                img_bytes, ct = _prepare_image_bytes(end_image)
                if img_bytes: url = _upload_media_to_fal(img_bytes, "end_img.png", ct); uploaded_media_urls[self.AUTO_KEY_END_IMAGE] = url;
                if not img_bytes or not url: upload_error = True; print(f"ERROR: {log_prefix()} End image prep/upload failed.")
            if input_video is not None and not upload_error:
                temp_vid_path = _save_tensor_to_temp_video(input_video, fps=output_video_fps)
                if temp_vid_path:
                    temp_files_to_clean.append(temp_vid_path)
                    with open(temp_vid_path, 'rb') as vf: video_bytes = vf.read()
                    if video_bytes: url = _upload_media_to_fal(video_bytes, os.path.basename(temp_vid_path), "video/mp4"); uploaded_media_urls[self.AUTO_KEY_INPUT_VIDEO] = url
                    if not video_bytes or not url: upload_error = True; print(f"ERROR: {log_prefix()} Input video read/upload failed.")
                else: upload_error = True; print(f"ERROR: {log_prefix()} Saving input video failed.")
            if input_audio is not None and not upload_error:
                temp_aud_path = _save_audio_tensor_to_temp_wav(input_audio)
                if temp_aud_path:
                    temp_files_to_clean.append(temp_aud_path)
                    with open(temp_aud_path, 'rb') as af: audio_bytes = af.read()
                    if audio_bytes: url = _upload_media_to_fal(audio_bytes, os.path.basename(temp_aud_path), "audio/wav"); uploaded_media_urls[self.AUTO_KEY_INPUT_AUDIO] = url
                    if not audio_bytes or not url: upload_error = True; print(f"ERROR: {log_prefix()} Input audio read/upload failed.")
                else: upload_error = True; print(f"ERROR: {log_prefix()} Saving input audio failed.")
        except Exception as e: print(f"ERROR: {log_prefix()} Media processing error: {e}"); traceback.print_exc(); upload_error = True
        if upload_error: print(f"ERROR: {log_prefix()} Aborting due to media errors."); # ... (cleanup upload temps) ...
            if cleanup_temp_files:
                 for tf in temp_files_to_clean:
                      if tf and os.path.exists(tf): try: os.remove(tf); except Exception: pass
            return (None,)

        # --- 3. Final Payload ---
        final_payload = user_params.copy()
        print(f"{log_prefix()} Injecting media URLs...")
        for auto_key, url in uploaded_media_urls.items():
            if auto_key in final_payload: print(f"WARN: {log_prefix()} Overwriting key '{auto_key}'.")
            final_payload[auto_key] = url
        print(f"{log_prefix()} Final Payload keys: {list(final_payload.keys())}")

        # --- 4. API Call & Processing ---
        try:
            print(f"{log_prefix()} Submitting job to: {endpoint_id}")
            if not endpoint_id or not endpoint_id.strip(): raise ValueError("Endpoint ID missing")
            handler = fal_client.submit(endpoint_id.strip(), arguments=final_payload); request_id = handler.request_id
            print(f"{log_prefix()} Job submitted. ID: {request_id}")
            response = _poll_fal_job(endpoint_id, request_id, timeout=900) # Poll
            print(f"{log_prefix()} Job {request_id} completed.")

            result_url = None; result_content_type = None; is_video = False; is_image = False
            if isinstance(response, dict): # Flexible result parsing
                vid_keys = ['video', 'videos']; img_keys = ['image', 'images']; url_key = 'url'
                for k in vid_keys:
                    item = response.get(k)
                    if isinstance(item, dict) and item.get(url_key): result_url=item[url_key]; result_content_type=item.get('content_type'); is_video=True; break
                    if isinstance(item, list) and len(item)>0 and isinstance(item[0],dict) and item[0].get(url_key): result_url=item[0][url_key]; result_content_type=item[0].get('content_type'); is_video=True; break
                if not result_url:
                     for k in img_keys:
                         item = response.get(k)
                         if isinstance(item, dict) and item.get(url_key): result_url=item[url_key]; result_content_type=item.get('content_type'); is_image=True; break
                         if isinstance(item, list) and len(item)>0 and isinstance(item[0],dict) and item[0].get(url_key): result_url=item[0][url_key]; result_content_type=item[0].get('content_type'); is_image=True; break
                if not result_url and response.get(url_key) and isinstance(response[url_key], str): result_url = response[url_key]; result_content_type = response.get('content_type')
                if result_url and not is_video and not is_image: # Guess type if needed
                    if result_content_type:
                        if 'video' in result_content_type: is_video=True
                        elif 'image' in result_content_type: is_image=True
                    if not is_video and not is_image:
                        if any(ext in result_url.lower() for ext in ['.mp4','.webm']): is_video=True
                        elif any(ext in result_url.lower() for ext in ['.png','.jpg','.jpeg','.webp']): is_image=True
            if not result_url: print(f"WARN: {log_prefix()} No media URL found in result."); return (None,)

            print(f"{log_prefix()} Downloading result: {result_url}")
            media_response = requests.get(result_url, stream=True, timeout=600); media_response.raise_for_status()

            if is_video:
                output_dir = folder_paths.get_temp_directory(); os.makedirs(output_dir, exist_ok=True)
                extension = '.mp4';
                if result_content_type and 'webm' in result_content_type: extension = '.webm'
                elif result_url.lower().endswith('.webm'): extension = '.webm'
                filename = f"fal_omni_result_vid_{uuid.uuid4().hex}{extension}"
                temp_download_filepath = os.path.join(output_dir, filename)
                temp_files_to_clean.append(temp_download_filepath) # Add download path for cleanup
                with open(temp_download_filepath, 'wb') as f:
                    for chunk in media_response.iter_content(chunk_size=1024*1024): f.write(chunk)
                print(f"{log_prefix()} Video downloaded: {temp_download_filepath}")
                print(f"{log_prefix()} Extracting frames...")
                frames_list = []; cap = cv2.VideoCapture(temp_download_filepath)
                if not cap.isOpened(): raise IOError(f"Cannot open video: {temp_download_filepath}")
                while True:
                    ret, frame = cap.read();
                    if not ret: break;
                    frames_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cap.release()
                if not frames_list: raise ValueError("No frames extracted")
                frames_np = np.stack(frames_list); frames_tensor = torch.from_numpy(frames_np).float()/255.0
                if frames_tensor is None: raise ValueError("Frame tensor failed")
                print(f"{log_prefix()} Video processed. Shape: {frames_tensor.shape}")
                return (frames_tensor,)
            elif is_image:
                image_bytes = media_response.content; pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                img_np = np.array(pil_image, dtype=np.float32) / 255.0; img_tensor = torch.from_numpy(img_np).unsqueeze(0)
                print(f"{log_prefix()} Image processed. Shape: {img_tensor.shape}")
                return (img_tensor,)
            else: print(f"ERROR: {log_prefix()} Could not determine result type."); return (None,)

      
        except KeyboardInterrupt:
            print(f"ERROR: {log_prefix()} Execution interrupted by user.")
            if request_id:
                print(f"{log_prefix()} Attempting to cancel Fal.ai job {request_id}...")
                try:
                    fal_client.cancel(endpoint_id, request_id)
                    print(f"{log_prefix()} Fal.ai cancel request sent for job {request_id}.")
                except Exception as cancel_e:
                    print(f"WARN: {log_prefix()} Failed to send cancel request: {cancel_e}")
            return (None,) # Use (None,) for I2V/T2V/Omni
        
        except TimeoutError as e:
            print(f"ERROR: {log_prefix()} Job timed out: {e}")
            if request_id:
                print(f"{log_prefix()} Attempting to cancel Fal.ai job {request_id} due to timeout...")
                try:
                    fal_client.cancel(endpoint_id, request_id)
                    print(f"{log_prefix()} Fal.ai cancel request sent for job {request_id}.")
                except Exception as cancel_e:
                    print(f"WARN: {log_prefix()} Failed to send cancel request after timeout: {cancel_e}")
            return (None,) # Use (None,) for I2V/T2V/Omni

    
        except RuntimeError as e: print(f"ERROR: {log_prefix()} Fal.ai job failed: {e}"); return (None,)
        except requests.exceptions.RequestException as e: print(f"ERROR: {log_prefix()} Network error: {e}"); traceback.print_exc(); return (None,)
        except (cv2.error, IOError, ValueError, Image.UnidentifiedImageError) as e: print(f"ERROR: {log_prefix()} Media processing error: {e}"); traceback.print_exc(); return (None,)
        except Exception as e: req_id_str=f"Req ID: {request_id}" if request_id else 'N/A'; print(f"ERROR: {log_prefix()} Unexpected error ({req_id_str}): {e}"); traceback.print_exc(); return (None,)
        finally:
            if cleanup_temp_files:
                print(f"{log_prefix()} Cleaning temporary files...")
                all_temp_files = temp_files_to_clean + ([temp_download_filepath] if temp_download_filepath else [])
                for temp_file in all_temp_files:
                    if temp_file and os.path.exists(temp_file):
                        try: print(f"{log_prefix()} Removing: {temp_file}"); os.remove(temp_file)
                        except Exception as e: print(f"WARN: {log_prefix()} Cleanup failed: {e}")
            else:
                all_temp_files = temp_files_to_clean + ([temp_download_filepath] if temp_download_filepath else [])
                if all_temp_files: print(f"{log_prefix()} Skipping cleanup for temp files: {all_temp_files}")


# --- Define the LipSync Node Class with Polling ---
class FalAILipSyncNode:
    # Constants
    PAYLOAD_KEY_VIDEO = "video_url"; PAYLOAD_KEY_AUDIO = "audio_url"; PAYLOAD_KEY_SYNC_MODE = "sync_mode"; PAYLOAD_KEY_MODEL = "model"
    ENDPOINT_V2 = "fal-ai/sync-lipsync/v2"; ENDPOINT_V1_9 = "fal-ai/sync-lipsync"
    MODEL_OPTIONS_V1_9 = ["lipsync-1.9.0-beta", "lipsync-1.8.0", "lipsync-1.7.1"]
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { "api_key": ("STRING", {"multiline": False, "default": "Paste FAL_KEY credentials here (e.g., key_id:key_secret)"}), "endpoint_version": (["v2.0", "v1.9"], {"default": "v2.0"}), "input_video": ("IMAGE",), "input_audio": ("AUDIO",), },
            "optional": { "sync_mode": (["cut_off", "loop", "bounce"], {"default": "cut_off"}), "model": (cls.MODEL_OPTIONS_V1_9, {"default": "lipsync-1.9.0-beta"}), "cleanup_temp_files": ("BOOLEAN", {"default": True}), "output_video_fps": ("INT", {"default": 30, "min": 1, "max": 120}), }
        }
    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("image_batch", "synced_audio", "video_path")
    FUNCTION = "execute_lipsync"
    CATEGORY = "BS_FalAi-API-Video"

    def execute_lipsync(self, api_key, endpoint_version, input_video, input_audio, sync_mode="cut_off", model="lipsync-1.9.0-beta", cleanup_temp_files=True, output_video_fps=30):
        def log_prefix(): return "FalAILipSyncNode:"
        print(f"{log_prefix()} Starting request for version: {endpoint_version}")

        # --- Setup & Initializations ---
        uploaded_media_urls = {}; temp_files_to_clean = []; payload = {}
        temp_download_filepath = None; frames_tensor = None; request_id = None

        # --- 1. Select Endpoint ---
        if endpoint_version == "v2.0": endpoint_to_call = self.ENDPOINT_V2
        elif endpoint_version == "v1.9": endpoint_to_call = self.ENDPOINT_V1_9
        else: print(f"ERROR: {log_prefix()} Invalid endpoint_version"); return (None, None, None)
        print(f"{log_prefix()} Targeting endpoint: {endpoint_to_call}")

        # --- 2. API Key & Input Validation ---
        if not api_key or not api_key.strip() or api_key == "Paste FAL_KEY credentials here (e.g., key_id:key_secret)": print(f"ERROR: {log_prefix()} API Key missing."); return (None, None, None)
        api_key_value = api_key.strip()
        try: os.environ["FAL_KEY"] = api_key_value; print(f"{log_prefix()} Using API Key.")
        except Exception as e: print(f"ERROR: {log_prefix()} Failed setting API Key: {e}"); traceback.print_exc(); return (None, None, None)
        if input_video is None: print(f"ERROR: {log_prefix()} 'input_video' is required."); return (None, None, None)
        if input_audio is None: print(f"ERROR: {log_prefix()} 'input_audio' is required."); return (None, None, None)

        # --- 3. Media Uploads ---
        upload_error = False
        try:
            print(f"{log_prefix()} Processing input_video...")
            temp_video_path = _save_tensor_to_temp_video(input_video, fps=output_video_fps)
            if temp_video_path and os.path.exists(temp_video_path):
                temp_files_to_clean.append(temp_video_path)
                with open(temp_video_path, 'rb') as vf: video_bytes = vf.read()
                if video_bytes: url = _upload_media_to_fal(video_bytes, "input_video.mp4", "video/mp4"); uploaded_media_urls[self.PAYLOAD_KEY_VIDEO] = url
                if not video_bytes or not url: upload_error = True; print(f"ERROR: {log_prefix()} Video read/upload failed.")
            else: upload_error = True; print(f"ERROR: {log_prefix()} Saving video tensor failed.")

            if not upload_error:
                print(f"{log_prefix()} Processing input_audio...")
                temp_audio_path = _save_audio_tensor_to_temp_wav(input_audio)
                if temp_audio_path and os.path.exists(temp_audio_path):
                    temp_files_to_clean.append(temp_audio_path)
                    with open(temp_audio_path, 'rb') as af: audio_bytes = af.read()
                    if audio_bytes: url = _upload_media_to_fal(audio_bytes, "input_audio.wav", "audio/wav"); uploaded_media_urls[self.PAYLOAD_KEY_AUDIO] = url
                    if not audio_bytes or not url: upload_error = True; print(f"ERROR: {log_prefix()} Audio read/upload failed.")
                else: upload_error = True; print(f"ERROR: {log_prefix()} Saving audio tensor failed.")
        except Exception as e: print(f"ERROR: {log_prefix()} Media processing error: {e}"); traceback.print_exc(); upload_error = True
        if upload_error: print(f"ERROR: {log_prefix()} Aborting due to media errors."); # ... (early cleanup) ...
            if cleanup_temp_files:
                for tf in temp_files_to_clean:
                     if tf and os.path.exists(tf): try: os.remove(tf); except Exception: pass
            return (None, None, None)

        # --- 4. Final Payload ---
        payload = { self.PAYLOAD_KEY_VIDEO: uploaded_media_urls[self.PAYLOAD_KEY_VIDEO], self.PAYLOAD_KEY_AUDIO: uploaded_media_urls[self.PAYLOAD_KEY_AUDIO], self.PAYLOAD_KEY_SYNC_MODE: sync_mode }
        if endpoint_version == "v1.9": payload[self.PAYLOAD_KEY_MODEL] = model; print(f"{log_prefix()} Adding 'model': {model} for v1.9.")
        elif model != self.MODEL_OPTIONS_V1_9[0]: print(f"WARN: {log_prefix()} 'model' param '{model}' ignored for v2.0.")
        print(f"{log_prefix()} Final Payload keys: {list(payload.keys())}")

        # --- 5. API Call & Processing ---
        try:
            print(f"{log_prefix()} Submitting job to: {endpoint_to_call}")
            handler = fal_client.submit(endpoint_to_call, arguments=payload); request_id = handler.request_id
            print(f"{log_prefix()} Job submitted. ID: {request_id}")
            response = _poll_fal_job(endpoint_to_call, request_id, timeout=900) # Poll
            print(f"{log_prefix()} Job {request_id} completed.")

            result_url = None; result_content_type = None
            if isinstance(response, dict) and 'video' in response and isinstance(response['video'], dict) and 'url' in response['video']:
                result_url = response['video']['url']; result_content_type = response['video'].get('content_type', 'video/mp4')
            else: raise ValueError(f"No 'video.url' in result: {json.dumps(response, indent=2)}")
            print(f"{log_prefix()} Result URL: {result_url}")

            print(f"{log_prefix()} Downloading result video...")
            media_response = requests.get(result_url, stream=True, timeout=600); media_response.raise_for_status()
            output_dir = folder_paths.get_temp_directory(); os.makedirs(output_dir, exist_ok=True)
            extension = '.mp4';
            if result_content_type and 'webm' in result_content_type: extension = '.webm'
            elif result_url.lower().endswith('.webm'): extension = '.webm'
            filename = f"fal_lipsync_result_{uuid.uuid4().hex}{extension}"
            temp_download_filepath = os.path.join(output_dir, filename)
            temp_files_to_clean.append(temp_download_filepath) # Add download path for cleanup
            with open(temp_download_filepath, 'wb') as f:
                for chunk in media_response.iter_content(chunk_size=1024*1024): f.write(chunk)
            print(f"{log_prefix()} Video downloaded: {temp_download_filepath}")

            print(f"{log_prefix()} Extracting frames...")
            frames_list = []; cap = cv2.VideoCapture(temp_download_filepath)
            if not cap.isOpened(): raise IOError(f"Cannot open video: {temp_download_filepath}")
            while True:
                ret, frame = cap.read();
                if not ret: break;
                frames_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            if not frames_list: raise ValueError(f"No frames extracted: {temp_download_filepath}")
            print(f"{log_prefix()} Extracted {len(frames_list)} frames.")
            frames_np = np.stack(frames_list, axis=0); frames_tensor = torch.from_numpy(frames_np).float() / 255.0

            if frames_tensor is None: raise ValueError("Frame tensor creation failed.")
            print(f"{log_prefix()} Completed. Shape: {frames_tensor.shape}")
            return (frames_tensor, input_audio, temp_download_filepath) # Success

        except KeyboardInterrupt:
            print(f"ERROR: {log_prefix()} Execution interrupted.")
            if request_id:
                print(f"{log_prefix()} Attempting cancel job {request_id}...")
                try: fal_client.cancel(endpoint_to_call, request_id) # Use endpoint_to_call here
                except Exception as ce: print(f"WARN: {log_prefix()} Cancel failed: {ce}")
            return (None, None, None) # Correct return for LipSync
        except TimeoutError as e:
            print(f"ERROR: {log_prefix()} Job timed out: {e}")
            if request_id:
                print(f"{log_prefix()} Attempting cancel job {request_id}...")
                try: fal_client.cancel(endpoint_to_call, request_id) # Use endpoint_to_call here
                except Exception as ce: print(f"WARN: {log_prefix()} Cancel failed: {ce}")
            return (None, None, None) # Correct return for LipSync
        except RuntimeError as e: print(f"ERROR: {log_prefix()} Fal.ai job failed: {e}"); return (None, None, None) # Correct return
        except requests.exceptions.RequestException as e: print(f"ERROR: {log_prefix()} Network error: {e}"); traceback.print_exc(); return (None, None, None)
        except (cv2.error, IOError, ValueError, Image.UnidentifiedImageError) as e: print(f"ERROR: {log_prefix()} Media processing error: {e}"); traceback.print_exc(); return (None, None, None)
        except Exception as e: req_id_str=f"Req ID: {request_id}" if request_id else 'N/A'; print(f"ERROR: {log_prefix()} Unexpected error ({req_id_str}): {e}"); traceback.print_exc(); return (None, None, None)
        finally:
            if cleanup_temp_files:
                print(f"{log_prefix()} Cleaning temporary files...")
                all_temp_files = temp_files_to_clean # temp_download_filepath is already in here if created
                for temp_file in all_temp_files:
                    if temp_file and os.path.exists(temp_file):
                        try: print(f"{log_prefix()} Removing: {temp_file}"); os.remove(temp_file)
                        except Exception as e: print(f"WARN: {log_prefix()} Cleanup failed for {temp_file}: {e}")
            else:
                all_temp_files = temp_files_to_clean
                if all_temp_files:
                     print(f"{log_prefix()} Skipping cleanup for temporary files:")
                     for tf in all_temp_files:
                          if tf and os.path.exists(tf): print(f" - {tf}")


# --- Node Registration ---
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
