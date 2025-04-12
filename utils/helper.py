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
import fal_client
from PIL import Image
import cv2
import folder_paths



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
    print(f"[Fal Helper] Started polling job {request_id} for endpoint {endpoint_id}...")
    print(f"[Fal Helper] Timeout set to {timeout}s, Interval {polling_interval}s.")

    while True:
        # --- Timeout Check ---
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
            print(f"ERROR: [Fal Helper] Job {request_id} exceeded timeout of {timeout} seconds.")
            # Attempt cancellation before raising timeout
            print(f"[Fal Helper] Attempting to cancel timed out job {request_id}...")
            try:
                # Use fal.cancel if available and correct
                # Assuming fal_client has a cancel method like this:
                fal_client.cancel(endpoint_id, request_id)  # Use the direct fal.cancel
                print(f"[Fal Helper] Cancel request sent for timed out job {request_id}.") # Updated log message
            except AttributeError:
                 print(f"WARN: [Fal Helper] fal_client.cancel not found. Cannot programmatically cancel job {request_id}.")
                 # If direct cancel isn't possible, maybe just raise timeout
            except Exception as cancel_e:
                print(f"WARN: [Fal Helper] Failed to send cancel request after timeout: {cancel_e}")
            raise TimeoutError(f"Fal.ai job {request_id} timed out after {timeout}s")

        # --- Status Check ---
        try:
            status_response = fal_client.status(endpoint_id, request_id)  # Use the direct fal.status
            status = status_response.status
            try:
              queue_pos = status_response.queue_position
            except AttributeError:
              queue_pos = None
            

            print(f"[Fal Helper] Job {request_id}: Status={status}, Queue={queue_pos if queue_pos is not None else 'N/A'}, Elapsed={elapsed_time:.1f}s")

            if status == "COMPLETED":
                print(f"[Fal Helper] Job {request_id} completed.") # No change in log message
                final_result = fal_client.result(endpoint_id, request_id)  # Use the direct fal.result
                return final_result

            elif status in ["ERROR", "FAILED", "CANCELLED"]:
                error_message = f"Fal.ai job {request_id} failed with status: {status}"
                print(f"ERROR: [Fal Helper] {error_message}")
                raise RuntimeError(error_message)
            
            elif status in ["IN_QUEUE", "IN_PROGRESS"]:
                try:
                    time.sleep(polling_interval)
                except KeyboardInterrupt:
                    print(f"WARN: [Fal Helper] KeyboardInterrupt caught during sleep for job {request_id}. Attempting cancellation...")
                    raise KeyboardInterrupt # Re-raise

            else: # Unknown status
                print(f"WARN: [Fal Helper] Job {request_id} has unknown status: {status}. Continuing poll.")
                try:
                    time.sleep(polling_interval)
                except KeyboardInterrupt:
                     print(f"WARN: [Fal Helper] KeyboardInterrupt during sleep (unknown status) for job {request_id}. Attempting cancellation...")
                     raise KeyboardInterrupt # Re-raise

        except KeyboardInterrupt: # Catch during API call itself
            print(f"WARN: [Fal Helper] KeyboardInterrupt caught during status check for job {request_id}. Attempting cancellation...")
            raise KeyboardInterrupt # Re-raise
        except requests.exceptions.RequestException as e:
             # Log network errors during status check but continue polling
             print(f"WARN: [Fal Helper] Network error checking status for job {request_id}: {e}. Retrying...")
             try:
                 time.sleep(polling_interval * 2) # Longer sleep after network error
             except KeyboardInterrupt:
                 print(f"WARN: [Fal Helper] KeyboardInterrupt during network error backoff for job {request_id}. Attempting cancellation...")
                 raise KeyboardInterrupt
        except Exception as e:
            # For other unexpected errors during status check, raise them
            print(f"ERROR: [Fal Helper] Unexpected error polling job {request_id}: {e}")
            traceback.print_exc()
            
            # Handle AttributeError if fal_client.status or fal_client.result not found
            if isinstance(e, AttributeError):
              if "status" in str(e) or "result" in str(e):
                  print(f"ERROR: [Fal Helper] fal_client.status or fal_client.result not found. Please ensure that the fal_client package is correctly installed and imported.")
                  raise
            else:
                print(f"ERROR: [Fal Helper] Unexpected error polling job {request_id}: {e}")
                traceback.print_exc()
                raise
            


# --- Helper Functions with Corrected Logging ---
def _prepare_image_bytes(image_tensor):
    if image_tensor is None: print("[Fal Helper] No image tensor provided."); return None, None
    print("[Fal Helper] Preparing image tensor...")
    try:
        if image_tensor.dim() == 4 and image_tensor.shape[0] == 1: img_tensor = image_tensor.squeeze(0)
        elif image_tensor.dim() == 3: img_tensor = image_tensor
        else: raise ValueError(f"Unexpected shape: {image_tensor.shape}")
        img_tensor = img_tensor.cpu(); img_np = img_tensor.numpy()
        if img_np.max() <= 1.0 and img_np.min() >= 0.0: img_np = (img_np * 255)
        img_np = img_np.astype(np.uint8); pil_image = Image.fromarray(img_np, 'RGB')
        buffered = io.BytesIO(); pil_image.save(buffered, format="PNG"); img_bytes = buffered.getvalue()
        print(f"[Fal Helper] Image tensor prep complete ({len(img_bytes)} bytes).")
        return img_bytes, "image/png"
    except Exception as e: print(f"ERROR: [Fal Helper] Image tensor processing failed: {e}"); traceback.print_exc(); return None, None

def _save_tensor_to_temp_video(image_tensor_batch, fps=30):
    if image_tensor_batch is None or image_tensor_batch.dim() != 4 or image_tensor_batch.shape[0] == 0: print("[Fal Helper] Invalid video tensor batch."); return None
    print("[Fal Helper] Saving video tensor batch to temp file...")
    temp_video_filepath = None # Ensure defined in outer scope
    video_writer = None # Ensure defined in outer scope
    try:
        batch_size, height, width, channels = image_tensor_batch.shape
        if channels != 3: print(f"ERROR: [Fal Helper] Expected 3 channels, got {channels}."); return None
        output_dir = folder_paths.get_temp_directory(); os.makedirs(output_dir, exist_ok=True)
        filename = f"fal_temp_upload_vid_{uuid.uuid4().hex}.mp4"
        temp_video_filepath = os.path.join(output_dir, filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v'); video_writer = cv2.VideoWriter(temp_video_filepath, fourcc, float(fps), (width, height))
        if not video_writer.isOpened(): print(f"ERROR: [Fal Helper] Failed to open video writer: {temp_video_filepath}"); return None

        image_tensor_batch_cpu = image_tensor_batch.cpu()
        for i in range(batch_size):
            frame_tensor = image_tensor_batch_cpu[i]; frame_np = frame_tensor.numpy()
            if frame_np.max() <= 1.0 and frame_np.min() >= 0.0: frame_np = (frame_np * 255)
            frame_np = frame_np.astype(np.uint8); frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
        print(f"[Fal Helper] Temp video saved: {temp_video_filepath}")
        return temp_video_filepath # Return path on success
    except Exception as e:
        print(f"ERROR: [Fal Helper] Video writing failed: {e}"); traceback.print_exc()
        # Cleanup partial file if error occurred AND path was defined
        if temp_video_filepath and os.path.exists(temp_video_filepath):
             try: os.remove(temp_video_filepath); print(f"[Fal Helper] Cleaned partial temp video: {temp_video_filepath}")
             except Exception as clean_e: print(f"WARN: [Fal Helper] Cleanup failed on video write error: {clean_e}")
        return None # Return None if exception occurred
    finally:
        if video_writer and video_writer.isOpened(): video_writer.release()


def _upload_media_to_fal(media_bytes, filename_hint, content_type):
    if not media_bytes: print(f"ERROR: [Fal Helper] No media bytes for upload ({filename_hint})."); return None
    temp_path = None
    try:
        temp_dir = folder_paths.get_temp_directory(); os.makedirs(temp_dir, exist_ok=True)
        ext = os.path.splitext(filename_hint)[1]
        if not ext and content_type:
             ct_map = {"png":".png", "jpeg":".jpg", "jpg":".jpg", "mp4":".mp4", "webm":".webm", "mp3":".mp3", "mpeg":".mp3", "wav":".wav"}
             for key, val in ct_map.items():
                 if key in content_type: ext = val; break
        if not ext: ext = ".tmp"
        temp_filename = f"fal_upload_{uuid.uuid4().hex}{ext}" ;temp_path = os.path.join(temp_dir, temp_filename)
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
    # print(f"[Fal Helper] _save_audio... received keys: {audio_data.keys() if isinstance(audio_data, dict) else 'Not dict'}") # Debug
    if not isinstance(audio_data, dict) or "sample_rate" not in audio_data or \
       ("samples" not in audio_data and "waveform" not in audio_data):
        print("ERROR: [Fal Helper] Invalid audio data format."); return None
    sample_rate = audio_data["sample_rate"]
    samples_tensor = audio_data.get("samples") or audio_data.get("waveform")
    if samples_tensor is None: print("ERROR: [Fal Helper] No audio tensor found."); return None
    print(f"[Fal Helper] Processing audio tensor (Rate: {sample_rate}, Shape: {samples_tensor.shape})")
    try:
        samples_tensor = samples_tensor.cpu();
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
