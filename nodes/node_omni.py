import fal_client
from PIL import Image
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
# NOTE: The following imports assume a specific file structure.
from ..utils.config import MODEL_CONFIGS, ALL_MODEL_NAMES_I2V, ALL_RESOLUTIONS, ALL_ASPECT_RATIOS
from ..utils.helper import _prepare_image_bytes, _save_tensor_to_temp_video, _upload_media_to_fal, _save_audio_tensor_to_temp_wav, _poll_fal_job

# --- Define the Omni Pro Node Class with Polling ---
class FalAPIOmniProNode:
    # Note: "input_image_urls" is no longer used for reference images, but kept for context.
    # The new keys "image_url" and "image_urls" are handled dynamically.
    AUTO_KEY_START_IMAGE = "image_url"; AUTO_KEY_END_IMAGE = "end_image_url"; AUTO_KEY_REFERENCE_IMAGES = "input_image_urls"
    AUTO_KEY_INPUT_VIDEO = "video_url"; AUTO_KEY_INPUT_AUDIO = "audio_url"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { "endpoint_id": ("STRING", {"multiline": False, "default": "fal-ai/some-model/endpoint-id"}), "api_key": ("STRING", {"multiline": False, "default": "Paste FAL_KEY credentials here (e.g., key_id:key_secret)"}), "parameters_json": ("STRING", {"multiline": True, "default": json.dumps({"prompt": "A description", "seed": 12345}, indent=2)}), },
            "optional": { "start_image": ("IMAGE",), "end_image": ("IMAGE",), "reference_images": ("IMAGE",), "input_video": ("IMAGE",), "input_audio": ("AUDIO",), "cleanup_temp_files": ("BOOLEAN", {"default": True}), "output_video_fps": ("INT", {"default": 30, "min": 1, "max": 120}), "override_start_image_param": ("STRING", {"multiline": False, "default": ""}), "override_end_image_param": ("STRING", {"multiline": False, "default": ""}), "override_video_param": ("STRING", {"multiline": False, "default": ""}), "override_audio_param": ("STRING", {"multiline": False, "default": ""}), }
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_batch",)
    FUNCTION = "execute_omni_request"
    CATEGORY = "BS_FalAi-API-Omni"

    def execute_omni_request(self, endpoint_id, api_key, parameters_json, start_image=None, end_image=None, reference_images=None, input_video=None, input_audio=None, cleanup_temp_files=True, output_video_fps=30, override_start_image_param="", override_end_image_param="", override_video_param="", override_audio_param=""):
        def log_prefix(): return "FalAPIOmniProNode:"
        print(f"{log_prefix()} Starting request...")

        # --- Setup & Initializations ---
        # NOTE: You'll need to provide the helper functions (_prepare_image_bytes, _upload_media_to_fal, etc.)
        # for this code to be fully runnable. The logic below assumes they exist.
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
            # The following section assumes helper functions like _prepare_image_bytes and _upload_media_to_fal are defined elsewhere
            if start_image is not None:
                img_bytes, ct = _prepare_image_bytes(start_image)
                url = _upload_media_to_fal(img_bytes, "start_img.png", ct) if img_bytes else None
                start_image_param_name = override_start_image_param.strip() if override_start_image_param.strip() else self.AUTO_KEY_START_IMAGE
                if url: uploaded_media_urls[start_image_param_name] = url
                else: upload_error = True; print(f"ERROR: {log_prefix()} Start image prep/upload failed.")
            if end_image is not None and not upload_error:
                img_bytes, ct = _prepare_image_bytes(end_image)
                url = _upload_media_to_fal(img_bytes, "end_img.png", ct) if img_bytes else None
                end_image_param_name = override_end_image_param.strip() if override_end_image_param.strip() else self.AUTO_KEY_END_IMAGE
                if url: uploaded_media_urls[end_image_param_name] = url
                else: upload_error = True; print(f"ERROR: {log_prefix()} End image prep/upload failed.")
            if input_video is not None and not upload_error:
                temp_vid_path = _save_tensor_to_temp_video(input_video, fps=output_video_fps)
                url = None
                if temp_vid_path:
                    temp_files_to_clean.append(temp_vid_path)
                    with open(temp_vid_path, 'rb') as vf: video_bytes = vf.read()
                    video_param_name = override_video_param.strip() if override_video_param.strip() else self.AUTO_KEY_INPUT_VIDEO
                    if video_bytes: url = _upload_media_to_fal(video_bytes, os.path.basename(temp_vid_path), "video/mp4")
                    if url: uploaded_media_urls[video_param_name] = url
                    else: upload_error = True; print(f"ERROR: {log_prefix()} Input video upload failed.")
                else: upload_error = True; print(f"ERROR: {log_prefix()} Saving input video failed.")
            if input_audio is not None and not upload_error:
                temp_aud_path = _save_audio_tensor_to_temp_wav(input_audio)
                url = None
                if temp_aud_path:
                    temp_files_to_clean.append(temp_aud_path)
                    with open(temp_aud_path, 'rb') as af: audio_bytes = af.read()
                    audio_param_name = override_audio_param.strip() if override_audio_param.strip() else self.AUTO_KEY_INPUT_AUDIO
                    if audio_bytes: url = _upload_media_to_fal(audio_bytes, os.path.basename(temp_aud_path), "audio/wav")
                    if url: uploaded_media_urls[audio_param_name] = url
                    else: upload_error = True; print(f"ERROR: {log_prefix()} Input audio upload failed.")
                else: upload_error = True; print(f"ERROR: {log_prefix()} Saving input audio failed.")
            
            # /-------------------------- MODIFIED SECTION START --------------------------/
            if reference_images is not None and not upload_error:
                reference_image_urls = []
                num_images_to_process = reference_images.shape[0]
                print(f"{log_prefix()} Processing {num_images_to_process} reference image(s)...")

                for i in range(num_images_to_process):
                    single_image_tensor = reference_images[i].unsqueeze(0)
                    img_bytes, ct = _prepare_image_bytes(single_image_tensor)
                    filename = f"ref_img_{uuid.uuid4().hex}.png"
                    url = None
                    if img_bytes:
                        print(f"{log_prefix()} Uploading reference image {i+1}/{num_images_to_process} ({filename})...")
                        url = _upload_media_to_fal(img_bytes, filename, ct)
                    
                    if url:
                        reference_image_urls.append(url)
                        print(f"{log_prefix()} Reference image {i+1} uploaded: {url}")
                    else:
                        upload_error = True
                        print(f"ERROR: {log_prefix()} Reference image {i+1} ({filename}) prep/upload failed.")
                        break
                
                if not upload_error and reference_image_urls:
                    # Conditional logic based on the number of successfully uploaded images
                    if len(reference_image_urls) == 1:
                        key = "image_url"
                        value = reference_image_urls[0] # The single URL string
                        # Warn user if this key is already in use (e.g., by start_image)
                        if key in uploaded_media_urls:
                            print(f"WARN: {log_prefix()} A single reference image is overwriting the '{key}' field.")
                        uploaded_media_urls[key] = value
                        print(f"{log_prefix()} Added 1 reference image to payload with key: '{key}'")
                    elif len(reference_image_urls) > 1:
                        key = "image_urls"
                        value = reference_image_urls # The list of URLs
                        uploaded_media_urls[key] = value
                        print(f"{log_prefix()} Added {len(value)} reference images to payload with key: '{key}'")

                elif not upload_error and not reference_image_urls and num_images_to_process > 0:
                    print(f"WARN: {log_prefix()} Reference images were provided but no URLs were generated.")
            # /--------------------------- MODIFIED SECTION END ---------------------------/

        except Exception as e: print(f"ERROR: {log_prefix()} Media processing error: {e}"); traceback.print_exc(); upload_error = True
        if upload_error: print(f"ERROR: {log_prefix()} Aborting due to media errors."); # ... (cleanup upload temps) ...

        
        if cleanup_temp_files:
            for tf in temp_files_to_clean:
                if tf and os.path.exists(tf):
                    try: os.remove(tf)
                    except Exception: pass

        
        # --- 3. Final Payload ---
        final_payload = user_params.copy()
        print(f"{log_prefix()} Injecting media URLs...")
        for auto_key, url_or_list in uploaded_media_urls.items():
            if auto_key in final_payload: print(f"WARN: {log_prefix()} Overwriting key '{auto_key}'.")
            final_payload[auto_key] = url_or_list
        print(f"{log_prefix()} Final Payload keys: {list(final_payload.keys())}")
        # print(f"{log_prefix()} Final Payload content: {json.dumps(final_payload, indent=2)}") # Uncomment for debugging

        # --- 4. API Call & Processing ---
        try:
            print(f"{log_prefix()} Submitting job to: {endpoint_id}")
            if not endpoint_id or not endpoint_id.strip(): raise ValueError("Endpoint ID missing")
            handler = fal_client.submit(endpoint_id.strip(), arguments=final_payload)
            request_id = handler.request_id
            print(f"{log_prefix()} Job submitted. ID: {request_id}")
            response = _poll_fal_job(endpoint_id, request_id, timeout=900) # Poll
            print(f"{log_prefix()} Job {request_id} completed.")

            result_url = None; result_content_type = None; is_video = False; is_image = False;

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
                        if 'video' in result_content_type.lower(): is_video=True
                        elif 'image' in result_content_type.lower(): is_image=True
                    if not is_video and not is_image: # Fallback guess by extension
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
            # ... (rest of the exception handling code is unchanged) ...
            print(f"ERROR: {log_prefix()} Execution interrupted by user."); 
            if request_id:
                print(f"{log_prefix()} Attempting to cancel Fal.ai job {request_id}...")
                try: fal_client.cancel(endpoint_id, request_id)
                except Exception as cancel_e: print(f"WARN: {log_prefix()} Failed to send cancel request: {cancel_e}")
            return (None,)
        except TimeoutError as e:
            print(f"ERROR: {log_prefix()} Job timed out: {e}"); 
            if request_id:
                print(f"{log_prefix()} Attempting to cancel Fal.ai job {request_id} due to timeout...")
                try: fal_client.cancel(endpoint_id, request_id)
                except Exception as cancel_e: print(f"WARN: {log_prefix()} Failed to send cancel request after timeout: {cancel_e}")
            return (None,)
        except RuntimeError as e: print(f"ERROR: {log_prefix()} Fal.ai job failed: {e}; req_id: {request_id if request_id else 'N/A'}"); return (None,)
        except requests.exceptions.RequestException as e: print(f"ERROR: {log_prefix()} Network error: {e}; req_id: {request_id if request_id else 'N/A'}"); traceback.print_exc(); return (None,)
        except (cv2.error, IOError, ValueError, Image.UnidentifiedImageError) as e: print(f"ERROR: {log_prefix()} Media processing error: {e}; req_id: {request_id if request_id else 'N/A'}"); traceback.print_exc(); return (None,)
        except Exception as e: 
            req_id_str = f"Req ID: {request_id}" if request_id else 'N/A'
            print(f"ERROR: {log_prefix()} Unexpected error ({req_id_str}): {e}")
            traceback.print_exc()
            return (None,)
