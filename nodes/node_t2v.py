import torch
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
import scipy.io.wavfile
import cv2 # Requires opencv-python: pip install opencv-python

# Helper to access ComfyUI's path functions
import folder_paths
from ..utils.config import MODEL_CONFIGS, ALL_MODEL_NAMES_T2V, ALL_RESOLUTIONS, ALL_ASPECT_RATIOS
from ..utils.helper import _prepare_image_bytes, _save_tensor_to_temp_video, _upload_media_to_fal, _save_audio_tensor_to_temp_wav, _poll_fal_job

# --- Define the Text-to-Video Node Class with Polling ---
class FalAPIVideoGeneratorT2V:
    @classmethod
    def INPUT_TYPES(cls):
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
            filename = f"fal_api_t2v_temp_{uuid.uuid4().hex}{extension}" # Use T2V prefix
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
            print(f"{log_prefix()} Execution interrupted by user.")
            if request_id:
                print(f"{log_prefix()} Attempting to cancel Fal.ai job {request_id}...")
                try:
                    fal_client.cancel(endpoint_id, request_id)
                    print(f"{log_prefix()} Fal.ai cancel request sent for job {request_id}.") # No change in log message
                except Exception as cancel_e:
                    print(f"WARN: {log_prefix()} Failed to send cancel request: {cancel_e}")
            return (None,)
        except TimeoutError as e:
            print(f"ERROR: {log_prefix()} Job timed out: {e}")
            if request_id:
                print(f"{log_prefix()} Attempting to cancel Fal.ai job {request_id} due to timeout...")
                try:
                    fal_client.cancel(endpoint_id, request_id)
                    print(f"{log_prefix()} Fal.ai cancel request sent for job {request_id}.") # No change in log message
                except Exception as cancel_e:
                    print(f"WARN: {log_prefix()} Failed to send cancel request after timeout: {cancel_e}")
            return (None,)
        except RuntimeError as e: print(f"ERROR: {log_prefix()} Fal.ai job failed: {e}"); return (None,)
        except requests.exceptions.RequestException as e: print(f"ERROR: {log_prefix()} Network error: {e}"); traceback.print_exc(); return (None,)
        except (cv2.error, IOError, ValueError, Image.UnidentifiedImageError) as e: print(f"ERROR: {log_prefix()} Media processing error: {e}"); traceback.print_exc(); return (None,)
        except Exception as e: req_id_str=f"Req ID: {request_id}" if request_id else 'N/A'; print(f"{log_prefix()} Unexpected error ({req_id_str}): {e}"); traceback.print_exc(); return (None,)
