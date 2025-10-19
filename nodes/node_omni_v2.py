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
import subprocess
import shutil

# Helper to access ComfyUI's path functions
import folder_paths
# NOTE: The following imports assume a specific file structure.
from ..utils.config import MODEL_CONFIGS, ALL_MODEL_NAMES_I2V, ALL_RESOLUTIONS, ALL_ASPECT_RATIOS
from ..utils.helper import _prepare_image_bytes, _save_tensor_to_temp_video, _upload_media_to_fal, _save_audio_tensor_to_temp_wav, _poll_fal_job

# --- Define the Omni Pro v2 Node Class with Polling ---
class FalOmniProV2Node:
    NODE_NAME = "Fal Omni Pro v2"
    NODE_DISPLAY_NAME = "Fal Omni Pro v2"

    AUTO_KEY_START_IMAGE = "image_url"; AUTO_KEY_END_IMAGE = "end_image_url"; AUTO_KEY_REFERENCE_IMAGES = "input_image_urls"
    AUTO_KEY_INPUT_VIDEO = "video_url"; AUTO_KEY_INPUT_AUDIO = "audio_url"
    
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "model_id": ("STRING", {"multiline": False, "default": "fal-ai/kling-video/v2.5-turbo/pro/image-to-video"}),
                "api_key": ("STRING", {"multiline": False, "default": os.getenv("FAL_KEY") or "Paste FAL_KEY credentials here (e.g., key_id:key_secret)"}),
            },
            "optional": {
                "start_image": ("IMAGE",),
                "end_image": ("IMAGE",),
                "reference_images": ("IMAGE",),
                "input_video": ("IMAGE",),
                "input_audio": ("AUDIO",),
                "cleanup_temp_files": ("BOOLEAN", {"default": True}),
                "output_video_fps": ("INT", {"default": 30, "min": 1, "max": 120}),
                "save_original_video": ("BOOLEAN", {"default": False}),
                "save_directory": ("STRING", {"default": "fal_omni_v2_output"}),
                "log_payload": ("BOOLEAN", {"default": False}),
            }
        }
        for i in range(1, 11):
            inputs["optional"][f"arg_{i}_name"] = ("STRING", {"default": "", "multiline": False})
            inputs["optional"][f"arg_{i}_value"] = ("STRING", {"default": "", "multiline": True})
        return inputs

    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT")
    RETURN_NAMES = ("image_batch", "audio", "fps")
    FUNCTION = "run"
    CATEGORY = "BS_FalAi-API-Omni"

    def parse_value(self, value):
        if not isinstance(value, str):
            return value
        
        clean_value = value.strip()
        
        if clean_value.lower() == 'true':
            return True
        if clean_value.lower() == 'false':
            return False
            
        try:
            return int(clean_value)
        except ValueError:
            pass
            
        try:
            return float(clean_value)
        except ValueError:
            pass
            
        return value

    def run(self, model_id, api_key, start_image=None, end_image=None, reference_images=None, input_video=None, input_audio=None, cleanup_temp_files=True, output_video_fps=30, **kwargs):
        save_original_video = kwargs.get("save_original_video", False)
        save_directory = kwargs.get("save_directory", "fal_omni_v2_output")
        log_payload = kwargs.get("log_payload", False)

        def log_prefix(): return "FalOmniProV2Node:"
        print(f"{log_prefix()} Starting request...")

        # --- Setup & Initializations ---
        uploaded_media_urls = {}; temp_files_to_clean = []; final_payload = {}
        temp_download_filepath = None; frames_tensor = None; img_tensor = None; audio_tensor = None
        request_id = None

        # --- 1. API Key & Params ---
        if not api_key or not api_key.strip() or api_key == "Paste FAL_KEY credentials here (e.g., key_id:key_secret)": print(f"ERROR: {log_prefix()} API Key missing."); return (None, None, 0.0)
        api_key_value = api_key.strip()
        try: os.environ["FAL_KEY"] = api_key_value; print(f"{log_prefix()} Using API Key.")
        except Exception as e: print(f"ERROR: {log_prefix()} Failed setting API Key: {e}"); traceback.print_exc(); return (None, None, 0.0)
        
        user_params = {}
        try:
            for i in range(1, 11):
                arg_name = kwargs.get(f"arg_{i}_name")
                arg_value = kwargs.get(f"arg_{i}_value")
                if arg_name and arg_name.strip():
                    user_params[arg_name.strip()] = self.parse_value(arg_value)
            print(f"{log_prefix()} Parsed dynamic parameters.")
        except Exception as e:
            print(f"ERROR: {log_prefix()} Failed to parse dynamic arguments: {e}")
            return (None, None, 0.0)

        # --- 2. Media Uploads ---
        upload_error = False
        try:
            if start_image is not None:
                img_bytes, ct = _prepare_image_bytes(start_image)
                url = _upload_media_to_fal(img_bytes, "start_img.png", ct) if img_bytes else None
                if url: uploaded_media_urls[self.AUTO_KEY_START_IMAGE] = url
                else: upload_error = True; print(f"ERROR: {log_prefix()} Start image prep/upload failed.")
            if end_image is not None and not upload_error:
                img_bytes, ct = _prepare_image_bytes(end_image)
                url = _upload_media_to_fal(img_bytes, "end_img.png", ct) if img_bytes else None
                if url: uploaded_media_urls[self.AUTO_KEY_END_IMAGE] = url
                else: upload_error = True; print(f"ERROR: {log_prefix()} End image prep/upload failed.")
            if input_video is not None and not upload_error:
                temp_vid_path = _save_tensor_to_temp_video(input_video, fps=output_video_fps)
                url = None
                if temp_vid_path:
                    temp_files_to_clean.append(temp_vid_path)
                    with open(temp_vid_path, 'rb') as vf: video_bytes = vf.read()
                    if video_bytes: url = _upload_media_to_fal(video_bytes, os.path.basename(temp_vid_path), "video/mp4")
                    if url: uploaded_media_urls[self.AUTO_KEY_INPUT_VIDEO] = url
                    else: upload_error = True; print(f"ERROR: {log_prefix()} Input video upload failed.")
                else: upload_error = True; print(f"ERROR: {log_prefix()} Saving input video failed.")
            if input_audio is not None and not upload_error:
                temp_aud_path = _save_audio_tensor_to_temp_wav(input_audio)
                url = None
                if temp_aud_path:
                    temp_files_to_clean.append(temp_aud_path)
                    with open(temp_aud_path, 'rb') as af: audio_bytes = af.read()
                    if audio_bytes: url = _upload_media_to_fal(audio_bytes, os.path.basename(temp_aud_path), "audio/wav")
                    if url: uploaded_media_urls[self.AUTO_KEY_INPUT_AUDIO] = url
                    else: upload_error = True; print(f"ERROR: {log_prefix()} Input audio upload failed.")
                else: upload_error = True; print(f"ERROR: {log_prefix()} Saving input audio failed.")
            
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
                    if len(reference_image_urls) == 1:
                        key = "image_url"
                        value = reference_image_urls[0]
                        if key in uploaded_media_urls:
                            print(f"WARN: {log_prefix()} A single reference image is overwriting the '{key}' field.")
                        uploaded_media_urls[key] = value
                        print(f"{log_prefix()} Added 1 reference image to payload with key: '{key}'")
                    elif len(reference_image_urls) > 1:
                        key = "image_urls"
                        value = reference_image_urls
                        uploaded_media_urls[key] = value
                        print(f"{log_prefix()} Added {len(value)} reference images to payload with key: '{key}'")

                elif not upload_error and not reference_image_urls and num_images_to_process > 0:
                    print(f"WARN: {log_prefix()} Reference images were provided but no URLs were generated.")

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
        
        if log_payload:
            print(f"{log_prefix()} Final Payload content: {json.dumps(final_payload, indent=2)}")

        # --- 4. API Call & Processing ---
        try:
            print(f"{log_prefix()} Submitting job to: {model_id}")
            if not model_id or not model_id.strip(): raise ValueError("Model ID missing")
            handler = fal_client.submit(model_id.strip(), arguments=final_payload)
            request_id = handler.request_id
            print(f"{log_prefix()} Job submitted. ID: {request_id}")
            response = _poll_fal_job(model_id, request_id, timeout=900) # Poll
            print(f"{log_prefix()} Job {request_id} completed.")

            result_url = None; result_content_type = None; is_video = False; is_image = False;

            if isinstance(response, dict):
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
                if result_url and not is_video and not is_image:
                    if result_content_type:
                        if 'video' in result_content_type.lower(): is_video=True
                        elif 'image' in result_content_type.lower(): is_image=True
                    if not is_video and not is_image:
                        if any(ext in result_url.lower() for ext in ['.mp4','.webm']): is_video=True
                        elif any(ext in result_url.lower() for ext in ['.png','.jpg','.jpeg','.webp']): is_image=True
            if not result_url: print(f"WARN: {log_prefix()} No media URL found in result."); return (None, None, 0.0)

            print(f"{log_prefix()} Downloading result: {result_url}")
            media_response = requests.get(result_url, stream=True, timeout=600); media_response.raise_for_status()

            if is_video:
                output_dir = folder_paths.get_temp_directory(); os.makedirs(output_dir, exist_ok=True)
                extension = '.mp4';
                if result_content_type and 'webm' in result_content_type: extension = '.webm'
                elif result_url.lower().endswith('.webm'): extension = '.webm'
                filename = f"fal_omni_result_vid_{uuid.uuid4().hex}{extension}"
                temp_download_filepath = os.path.join(output_dir, filename)
                temp_files_to_clean.append(temp_download_filepath)
                with open(temp_download_filepath, 'wb') as f:
                    for chunk in media_response.iter_content(chunk_size=1024*1024): f.write(chunk)
                print(f"{log_prefix()} Video downloaded: {temp_download_filepath}")

                if save_original_video:
                    try:
                        final_output_dir = os.path.join(folder_paths.get_output_directory(), save_directory)
                        os.makedirs(final_output_dir, exist_ok=True)
                        final_filename = f"fal_omni_{int(time.time())}_{uuid.uuid4().hex[:8]}{extension}"
                        final_filepath = os.path.join(final_output_dir, final_filename)
                        shutil.copy(temp_download_filepath, final_filepath)
                        print(f"{log_prefix()} Saved original video to: {final_filepath}")
                    except Exception as e:
                        print(f"WARN: {log_prefix()} Could not save original video: {e}")

                try:
                    temp_wav_path = os.path.splitext(temp_download_filepath)[0] + ".wav"
                    ffmpeg_cmd = ['ffmpeg', '-i', temp_download_filepath, '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', temp_wav_path]
                    
                    print(f"{log_prefix()} Attempting to extract audio with ffmpeg...")
                    subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    if os.path.exists(temp_wav_path):
                        temp_files_to_clean.append(temp_wav_path)
                        sample_rate, wav_data = scipy.io.wavfile.read(temp_wav_path)
                        
                        # Normalize and convert to tensor
                        normalized_wav = wav_data.astype(np.float32) / np.iinfo(wav_data.dtype).max
                        
                        if normalized_wav.ndim > 1:
                            waveform_np = normalized_wav.T
                        else:
                            waveform_np = normalized_wav

                        waveform = torch.from_numpy(waveform_np).unsqueeze(0)
                        audio_tensor = {"waveform": waveform, "sample_rate": sample_rate}
                        print(f"{log_prefix()} Audio extracted successfully. Shape: {waveform.shape}")
                    else:
                        print(f"WARN: {log_prefix()} ffmpeg ran but output file not found. Video may have no audio.")

                except FileNotFoundError:
                    print(f"WARN: {log_prefix()} `ffmpeg` not found. Cannot extract audio. Please install ffmpeg and ensure it's in your system's PATH.")
                except subprocess.CalledProcessError:
                    print(f"WARN: {log_prefix()} ffmpeg failed. The video likely has no audio stream.")
                except Exception as e:
                    print(f"WARN: {log_prefix()} An unexpected error occurred during audio extraction: {e}")

                print(f"{log_prefix()} Extracting frames and FPS...")
                frames_list = []; cap = cv2.VideoCapture(temp_download_filepath)

                detected_fps = cap.get(cv2.CAP_PROP_FPS)
                if detected_fps == 0:
                    print(f"WARN: {log_prefix()} Detected FPS is 0. Using input FPS value as fallback.")
                    detected_fps = float(output_video_fps)

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
                return (frames_tensor, audio_tensor, detected_fps)
            elif is_image:
                image_bytes = media_response.content; pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                img_np = np.array(pil_image, dtype=np.float32) / 255.0; img_tensor = torch.from_numpy(img_np).unsqueeze(0)
                print(f"{log_prefix()} Image processed. Shape: {img_tensor.shape}")
                return (img_tensor, None, 0.0)
            else: print(f"ERROR: {log_prefix()} Could not determine result type."); return (None, None, 0.0)

        except KeyboardInterrupt:
            print(f"ERROR: {log_prefix()} Execution interrupted by user."); 
            if request_id:
                print(f"{log_prefix()} Attempting to cancel Fal.ai job {request_id}...")
                try: fal_client.cancel(model_id, request_id)
                except Exception as cancel_e: print(f"WARN: {log_prefix()} Failed to send cancel request: {cancel_e}")
            return (None, None, 0.0)
        except TimeoutError as e:
            print(f"ERROR: {log_prefix()} Job timed out: {e}"); 
            if request_id:
                print(f"{log_prefix()} Attempting to cancel Fal.ai job {request_id} due to timeout...")
                try: fal_client.cancel(model_id, request_id)
                except Exception as cancel_e: print(f"WARN: {log_prefix()} Failed to send cancel request after timeout: {cancel_e}")
            return (None, None, 0.0)
        except RuntimeError as e: print(f"ERROR: {log_prefix()} Fal.ai job failed: {e}; req_id: {request_id if request_id else 'N/A'}"); return (None, None, 0.0)
        except requests.exceptions.RequestException as e: print(f"ERROR: {log_prefix()} Network error: {e}; req_id: {request_id if request_id else 'N/A'}"); traceback.print_exc(); return (None, None, 0.0)
        except (cv2.error, IOError, ValueError, Image.UnidentifiedImageError) as e: print(f"ERROR: {log_prefix()} Media processing error: {e}; req_id: {request_id if request_id else 'N/A'}"); traceback.print_exc(); return (None, None, 0.0)
        except Exception as e: 
            req_id_str = f"Req ID: {request_id}" if request_id else 'N/A'
            print(f"ERROR: {log_prefix()} Unexpected error ({req_id_str}): {e}")
            traceback.print_exc()
            return (None, None, 0.0)
