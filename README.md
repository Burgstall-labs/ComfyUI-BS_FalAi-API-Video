# ComfyUI Fal.ai API Nodes (ComfyUI-BS_FalAi-API-Video)

This repository contains custom nodes for ComfyUI that allow you to interact with the [fal.ai](https://fal.ai/) API, enabling the use of various AI models directly within your ComfyUI workflows. Currently, it focuses on video generation models but also includes a flexible "Omni" node for interacting with potentially any fal.ai endpoint.

## Nodes Included

1.  **FAL AI Text-to-Video (`FalAPIVideoGeneratorT2V`)**:
    *   Generates videos based on text prompts.
    *   Uses a predefined list of known fal.ai text-to-video model endpoints.
    *   Provides standard controls for parameters like seed, steps, duration, resolution, aspect ratio, etc., via dedicated input fields.

2.  **FAL AI Image-to-Video (`FalAPIVideoGeneratorI2V`)**:
    *   Generates videos based on an initial input image and a text prompt.
    *   Uses a predefined list of known fal.ai image-to-video model endpoints.
    *   Provides standard controls similar to the T2V node, plus an image input socket.

3.  **fal.ai API Omni Pro Node (`FalAPIOmniProNode`)**:
    *   A highly flexible node designed to interact with **any** fal.ai model endpoint.
    *   You provide the specific `endpoint_id` as text input.
    *   Handles **automatic uploading** of media inputs (start image, end image, input video, input audio) if connected.
    *   Requires other model parameters (like `seed`, `steps`, `prompt`, endpoint-specific booleans, enums, etc.) to be provided as a **JSON object** in a text field. Please refer to each corresponding endpoint API description, like https://fal.ai/models/fal-ai/kling-video/v1/standard/image-to-video/api
    *   Returns the resulting image or video frames as a standard ComfyUI IMAGE batch.

## Features

*   Integrate powerful fal.ai models into ComfyUI.
*   Dedicated nodes for common Text-to-Video and Image-to-Video tasks with easy controls.
*   Versatile "Omni Pro" node to access virtually any fal.ai endpoint.
*   Automatic handling of image, video, and audio uploads for the Omni Pro node.
*   Leverages the `fal-client` library for API interaction.

## Installation

1.  **Clone the Repository:**
    Navigate to your ComfyUI `custom_nodes` directory and clone this repository:
    ```bash
    cd ComfyUI/custom_nodes/
    git clone <your-repo-url> ComfyUI-BS_FalAi-API-Video
    ```
    (Replace `<your-repo-url>` with the actual URL of your GitHub repository)

2.  **Install Dependencies:**
    Navigate into the cloned directory and install the required Python packages using pip. It's recommended to activate your ComfyUI's virtual environment first.
    ```bash
    cd ComfyUI-BS_FalAi-API-Video
    pip install -r requirements.txt
    ```
    Alternatively, if `requirements.txt` is not provided, install manually:
    ```bash
    pip install fal-client requests numpy Pillow opencv-python scipy
    ```

3.  **Restart ComfyUI:** Stop your ComfyUI instance completely and restart it. The new nodes should appear under the "BS_FalAi-API-Video" and "BS_FalAi-API-Omni" categories (or however you've set the `CATEGORY` in the code).

## Usage

### Prerequisites

*   **Fal.ai Account:** You need an account with [fal.ai](https://fal.ai/).
*   **API Key:** Obtain your API key credentials from your fal.ai dashboard. It typically looks like `key_id:key_secret`.

### All Nodes

*   **API Key Input:** Each node requires your fal.ai API key. Paste your `key_id:key_secret` string into the `api_key` text field on the node.

### Text-to-Video & Image-to-Video Nodes

1.  Add the `FAL AI Text-to-Video` or `FAL AI Image-to-Video` node to your workflow.
2.  Select the desired fal.ai `model_name` from the dropdown list.
3.  Enter your `api_key`.
4.  For I2V, connect an input `image` (standard ComfyUI IMAGE type).
5.  Fill in the `prompt` and optionally the `negative_prompt`.
6.  Adjust other parameters like `seed`, `steps`, `duration_seconds`, `resolution_enum`, `aspect_ratio_enum`, etc., using the provided widgets.
7.  The node will call the selected fal.ai endpoint and output the generated video frames as an IMAGE batch.

### fal.ai API Omni Pro Node

This node offers maximum flexibility but requires you to know the specifics of the fal.ai endpoint you want to use.

1.  Add the `fal.ai API Omni Pro Node` to your workflow.
2.  **`endpoint_id`**: Paste the **exact** endpoint ID string from fal.ai documentation (e.g., `fal-ai/stable-diffusion-xl`, `fal-ai/kling-video/lipsync/audio-to-video`).
3.  **`api_key`**: Enter your `key_id:key_secret`.
4.  **`parameters_json`**: This is critical.
    *   Provide a **valid JSON object** containing **only the non-media parameters** required or accepted by the specific `endpoint_id`.
    *   Refer to the fal.ai documentation for that endpoint to see what parameters it needs (e.g., `prompt`, `seed`, `num_inference_steps`, custom booleans, styles, etc.).
    *   **DO NOT** include keys for media inputs (`image_url`, `video_url`, `audio_url`, `end_image_url`) in this JSON if you are connecting the corresponding input sockets below. The node handles those automatically.
    *   Example JSON:
        ```json
        {
          "prompt": "A photorealistic cat wearing sunglasses",
          "seed": 42,
          "num_inference_steps": 30,
          "guidance_scale": 7.5
        }
        ```
5.  **Media Inputs (Optional)**:
    *   `start_image` (IMAGE): Connect if the endpoint needs an initial image.
    *   `end_image` (IMAGE): Connect if the endpoint needs a secondary/end image.
    *   `input_video` (IMAGE batch): Connect video frames if the endpoint needs an input video.
    *   `input_audio` (AUDIO): Connect audio data if the endpoint needs audio input.
    *   **Automatic Handling:** If you connect any of these inputs, the node will automatically:
        *   Prepare the data (convert tensors, save to temporary files).
        *   Upload the data to fal.ai using `fal_client.upload_file`.
        *   Inject the resulting fal.ai URL into the final API request payload using standard keys:
            *   `start_image` -> uses key `"image_url"`
            *   `end_image` -> uses key `"end_image_url"`
            *   `input_video` -> uses key `"video_url"`
            *   `input_audio` -> uses key `"audio_url"`
        *   **Warning:** If you manually specify one of these keys (e.g., `"image_url"`) in your `parameters_json`, the automatically uploaded file's URL will **overwrite** your manual value.
6.  **Other Inputs**: Adjust `cleanup_temp_files` and `output_video_fps` (used for converting input video tensors) if needed.
7.  The node calls the specified `endpoint_id` with the combined parameters (from JSON + auto-injected media URLs) and returns the result as an IMAGE batch.

## Dependencies

*   `fal-client`
*   `requests`
*   `numpy`
*   `Pillow`
*   `opencv-python`
*   `scipy`

(A `requirements.txt` file is recommended for easy installation.)

## Notes & Troubleshooting

*   **API Key Format:** Ensure your API key is in the correct `key_id:key_secret` format. An incorrect key will lead to authentication errors.
*   **Endpoint ID:** Double-check the `endpoint_id` string for the Omni Pro node. Typos or incorrect IDs will cause "Not Found" errors.
*   **JSON Validity:** Ensure the text entered into `parameters_json` is valid JSON. Use an online JSON validator if unsure. Syntax errors will prevent the node from running.
*   **Endpoint Parameters:** For the Omni Pro node, always consult the specific fal.ai endpoint documentation to understand the required and optional parameters and their expected data types (string, integer, boolean, etc.). Pass these in the `parameters_json` field.
*   **Temporary Files:** The Omni Pro node creates temporary files for media uploads and downloads. Ensure ComfyUI has write permissions to its temporary directory (`ComfyUI/temp`). If `cleanup_temp_files` is disabled, these files will remain after execution.

## License

(Consider adding a license, e.g., MIT)
