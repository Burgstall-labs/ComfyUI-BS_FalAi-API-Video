# ComfyUI Fal.ai API Nodes (ComfyUI-BS_FalAi-API-Video)

This repository contains custom nodes for ComfyUI that allow you to interact with the [fal.ai](https://fal.ai/) API, enabling the use of various AI models directly within your ComfyUI workflows. It includes nodes for specific tasks like video generation and lipsyncing, as well as a flexible "Omni" node for interacting with potentially any fal.ai endpoint.

## Project Structure

The project has a clean and standard structure for ComfyUI custom nodes:
-   `ComfyUI-BS_FalAi-API-Video/`: Main directory.
    -   `nodes/`: Contains all node implementation files (`node_i2v.py`, `node_t2v.py`, `node_omni_v2.py`, etc.).
    -   `utils/`: Contains shared utility modules like `config.py` and `helper.py`.
    -   `__init__.py`: The main entry point that registers all nodes with ComfyUI.
    -   `README.md`: This file.

## Nodes Included

1.  **FAL AI Text-to-Video (`FalAPIVideoGeneratorT2V`)**:
    *   Generates videos based on text prompts using predefined fal.ai text-to-video endpoints.
    *   Provides standard controls via dedicated input fields (seed, steps, duration, resolution, etc.).

2.  **FAL AI Image-to-Video (`FalAPIVideoGeneratorI2V`)**:
    *   Generates videos based on an initial image and text prompt using predefined fal.ai image-to-video endpoints.
    *   Includes standard controls plus an image input socket.

3.  **FAL AI LipSync Node (`FalAILipSyncNode`)**:
    *   Performs lipsyncing on an input video using an input audio track.
    *   Supports both `v1.9` and `v2.0` versions of the fal.ai lipsync endpoint via a selector.
    *   Includes specific controls like `sync_mode` and (for v1.9) `model` selection.

4.  **FAL AI API Omni Pro Node (`FalAPIOmniProNode`)**:
    *   A highly flexible node to interact with **any** fal.ai model endpoint.
    *   Requires the specific `endpoint_id` as text input.
    *   Handles **automatic uploading** of media inputs (start image, end image, input video, input audio).
    *   Requires other non-media model parameters to be provided as a **JSON object** in the `parameters_json` field. Refer to the specific endpoint documentation on fal.ai for required parameters (e.g., the [Kling Image-to-Video API docs](https://fal.ai/models/fal-ai/kling-video/v1/standard/image-to-video/api)).
    *   Returns the resulting image or video frames.

5.  **FAL AI Omni Pro v2 (`FalOmniProV2Node`)**:
    *   The new **future-proof** powerhouse node designed to work with **any current or future `fal.ai` API endpoint** without requiring code changes.
    *   Instead of a JSON blob, it uses dynamic `arg_name` and `arg_value` fields, allowing you to build the API payload directly in the UI.
    *   **Automatic Media Uploads**: Handles `start_image`, `end_image`, `reference_images`, `input_video`, and `input_audio` sockets automatically.
    *   **Batch Image Support**: Intelligently uses `image_url` for a single reference image and `image_urls` for a batch of reference images.
    *   **Multiple Outputs**:
        *   `image_batch`: The generated video frames or image.
        *   `audio`: Automatically extracts the audio from the generated video, if present.
        *   `fps`: Outputs the detected frames per second of the generated video.
    *   **Save Original Video**: Includes a toggle to save the pristine, untouched video file from the API directly to your ComfyUI output directory.
    *   **Debug Logging**: A `log_payload` toggle to print the exact JSON payload sent to the API in the console, for easy debugging.

## Features

*   Integrate powerful fal.ai models into ComfyUI.
*   Dedicated nodes for common Text-to-Video, Image-to-Video, and LipSync tasks.
*   Versatile "Omni Pro" node for accessing virtually any fal.ai endpoint.
*   A **future-proof "Omni Pro v2" node** with dynamic argument building, audio extraction, and more.
*   Automatic handling of media uploads for relevant nodes.
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
    Ensure you have activated your ComfyUI's Python virtual environment. Navigate into the cloned directory and install requirements:
    ```bash
    cd ComfyUI-BS_FalAi-API-Video
    pip install -r requirements.txt
    ```

3.  **Restart ComfyUI:** Stop your ComfyUI instance completely and restart it. The new nodes should appear under the "BS_FalAi-API-Video" and "BS_FalAi-API-Omni" categories (or your configured category).

## Usage

### Prerequisites

*   **Fal.ai Account:** You need an account with [fal.ai](https://fal.ai/).
*   **API Key:** Obtain your API key credentials from your fal.ai dashboard. It typically looks like `key_id:key_secret`.

### All Nodes

*   **API Key Input:** Each node requires your fal.ai API key. Paste your `key_id:key_secret` string into the `api_key` text field on the node.

### Omni Pro v2 Node (Recommended)

This is the most flexible and powerful node.

1.  Add the `Fal Omni Pro v2` node to your workflow.
2.  **`model_id`**: Paste the exact endpoint ID from the `fal.ai` model page (e.g., `fal-ai/veo3.1/reference-to-video`).
3.  **`api_key`**: Enter your `key_id:key_secret`.
4.  **Dynamic Arguments (`arg_1_name`, `arg_1_value`, etc.)**:
    *   For each parameter required by the API (like `prompt`, `resolution`, `generate_audio`), fill in a pair of fields.
    *   `arg_1_name`: `prompt`
    *   `arg_1_value`: `A beautiful cinematic shot of a sunset.`
    *   The node automatically handles converting values like `true`, `false`, and numbers to the correct type.
5.  **Media Inputs**: Connect any required media, such as `reference_images`.
6.  **Save & FPS**:
    *   Enable the `save_original_video` toggle to keep a copy of the output MP4.
    *   Connect the `fps` and `audio` outputs to other nodes (like VHS VideoCombine) as needed.
7.  **Debugging**: Enable `log_payload` to see the final JSON payload in your console.

### Legacy Nodes (Text-to-Video, Image-to-Video, etc.)

These are simpler to use for their specific tasks but less flexible.

1.  Add the desired node (e.g., `FAL AI Text-to-Video`).
2.  Select the `model_name` from the dropdown.
3.  Fill in your `api_key` and other parameters.
4.  Connect media inputs if required.
5.  The output will be the generated video frames.

## Dependencies

*   `fal-client`
*   `requests`
*   `numpy`
*   `Pillow`
*   `opencv-python`
*   `scipy`
*   **`ffmpeg`**: Required by the Omni Pro v2 node for audio extraction. Please ensure `ffmpeg` is installed and accessible in your system's PATH.

(A `requirements.txt` file is included for easy installation via `pip install -r requirements.txt`.)

## Notes & Troubleshooting

*   **FFMPEG for Omni Pro v2**: The audio extraction feature of the v2 node will fail gracefully if `ffmpeg` is not found, but you will not get an audio output.
*   **API Key Format:** Ensure the key is `key_id:key_secret`.
*   **Endpoint ID/Parameters:** For the Omni nodes, always check the official fal.ai documentation for the correct endpoint ID and its specific parameter requirements.
*   **JSON Validity:** Use a validator to check your JSON in the legacy Omni node's `parameters_json` field if you encounter issues.
*   **Temporary Files:** Ensure write permissions for ComfyUI's temp directory. Disable `cleanup_temp_files` only for debugging.

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.
