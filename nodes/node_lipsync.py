from PIL import Image
import fal_client
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
from ..utils.config import MODEL_CONFIGS, ALL_MODEL_NAMES_I2V, ALL_RESOLUTIONS, ALL_ASPECT_RATIOS
from ..utils.helper import _prepare_image_bytes, _save_tensor_to_temp_video, _upload_media_to_fal, _save_audio_tensor_to_temp_wav, _poll_fal_job

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
            url_vid = None
            if temp_video_path and os.path.exists(temp_video_path):
                temp_files_to_clean.append(temp_video_path)
                with open(temp_video_path, 'rb') as vf: video_bytes = vf.read()
                if video_bytes: url_vid = _upload_media_to_fal(video_bytes, "input_video.mp4", "video/mp4")
                if url_vid: uploaded_media_urls[self.PAYLOAD_KEY_VIDEO] = url_vid
                else: upload_error = True; print(f"ERROR: {log_prefix()} Video read/upload failed.")
            else: upload_error = True; print(f"ERROR: {log_prefix()} Saving video tensor failed.")

            if not upload_error:
                print(f"{log_prefix()} Processing input_audio...")
                temp_audio_path = _save_audio_tensor_to_temp_wav(input_audio)
                url_aud = None
                if temp_audio_path and os.path.exists(temp_audio_path):
                    temp_files_to_clean.append(temp_audio_path)
                    with open(temp_audio_path, 'rb') as af: audio_bytes = af.read()
                    if audio_bytes: url_aud = _upload_media_to_fal(audio_bytes, "input_audio.wav", "audio/wav")
                    if url_aud: uploaded_media_urls[self.PAYLOAD_KEY_AUDIO] = url_aud
                    else: upload_error = True; print(f"ERROR: {log_prefix()} Audio read/upload failed.")
                else: upload_error = True; print(f"ERROR: {log_prefix()} Saving audio tensor failed.")
        except Exception as e: print(f"ERROR: {log_prefix()} Media processing error: {e}"); traceback.print_exc(); upload_error = True
        if upload_error: print(f"ERROR: {log_prefix()} Aborting due to media errors.");
        if cleanup_temp_files:
            for tf in temp_files_to_clean:
                if tf and os.path.exists(tf):
                    try: os.remove(tf)
                    except Exception: pass
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
            frames_np = np.stack(frames_list); frames_tensor = torch.from_numpy(frames_np).float() / 255.0

            if frames_tensor is None: raise ValueError("Frame tensor creation failed.")
            print(f"{log_prefix()} Completed. Shape: {frames_tensor.shape}")
            return (frames_tensor, input_audio, temp_download_filepath) # Success

        except KeyboardInterrupt:
            print(f"ERROR: {log_prefix()} Execution interrupted by user.")
            if request_id:
                print(f"{log_prefix()} Attempting to cancel Fal.ai job {request_id}...")
                try:
                    fal_client.cancel(endpoint_to_call, request_id) # Use endpoint_to_call here
                    print(f"{log_prefix()} Fal.ai cancel request sent for job {request_id}.")
                except Exception as cancel_e:
                    print(f"WARN: {log_prefix()} Failed to send cancel request: {cancel_e}")
            return (None, None, None) # Correct return for LipSync
        except TimeoutError as e:
            print(f"ERROR: {log_prefix()} Job timed out: {e}")
            if request_id:
                print(f"{log_prefix()} Attempting to cancel Fal.ai job {request_id} due to timeout...")
                try:
                    fal_client.cancel(endpoint_to_call, request_id) # Use endpoint_to_call here
                    print(f"{log_prefix()} Fal.ai cancel request sent for job {request_id}.")
                except Exception as cancel_e:
                    print(f"WARN: {log_prefix()} Failed to send cancel request after timeout: {cancel_e}")
            return (None, None, None) # Correct return for LipSync
        except RuntimeError as e: print(f"ERROR: {log_prefix()} Fal.ai job failed: {e}"); return (None, None, None) # Correct return
        except requests.exceptions.RequestException as e: print(f"ERROR: {log_prefix()} Network error: {e}"); traceback.print_exc(); return (None, None, None)
        except (cv2.error, IOError, ValueError, Image.UnidentifiedImageError) as e: print(f"ERROR: {log_prefix()} Media processing error: {e}"); traceback.print_exc(); return (None, None, None)
        except Exception as e: req_id_str=f"Req ID: {request_id}" if request_id else 'N/A'; print(f"ERROR: {log_prefix()} Unexpected error ({req_id_str}): {e}"); traceback.print_exc(); return (None, None, None)