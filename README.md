# ComfyUI Fal.ai API Nodes (ComfyUI-BS_FalAi-API-Video)

This repository contains custom nodes for ComfyUI that allow you to interact with the [fal.ai](https://fal.ai/) API, enabling the use of various AI models directly within your ComfyUI workflows. It includes nodes for specific tasks like video generation and lipsyncing, as well as a flexible "Omni" node for interacting with potentially any fal.ai endpoint.

## Project Structure

The project has been refactored to improve organization. The main components are now distributed across multiple files:

-   `ComfyUI-BS_FalAi-API-Video/`: Main directory
    -   `utils/`: Contains utility modules.
        -   `config.py`: Model configurations.
        -   `helper.py`: Helper functions.
-   `nodes/`: Contains the node implementations.
    -   `node_i2v.py`: Image-to-Video node.
    -   `node_t2v.py`: Text-to-Video node.
    -   `node_omni.py`: Omni Pro node.
    -   `node_lipsync.py`: LipSync node.
    -   `__init__.py`: Marks the nodes directory as a package.
    -   `main.py`: Contains the `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS`.
    -   `__init__.py`: Exports the `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS`

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

## Features

*   Integrate powerful fal.ai models into ComfyUI.
*   Dedicated nodes for common Text-to-Video, Image-to-Video, and LipSync tasks.
*   Versatile "Omni Pro" node for accessing virtually any fal.ai endpoint.
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

### Text-to-Video & Image-to-Video Nodes

1.  Add the `FAL AI Text-to-Video` or `FAL AI Image-to-Video` node.
2.  Select the desired fal.ai `model_name` from the dropdown.
3.  Enter your `api_key`.
4.  For I2V, connect an input `image`.
5.  Fill in the `prompt` and optionally `negative_prompt`.
6.  Adjust other parameters using the provided widgets.
7.  Output: Generated video frames (IMAGE batch).

### FAL AI LipSync Node

1.  Add the `FAL AI LipSync Node`.
2.  Enter your `api_key`.
3.  Select the `endpoint_version` ("v2.0" or "v1.9").
4.  Connect the required `input_video` (IMAGE batch) and `input_audio` (AUDIO).
5.  Optionally, adjust the `sync_mode`.
6.  If using "v1.9", optionally select a specific `model` version. The `model` setting is ignored if "v2.0" is selected.
7.  Output: Lipsynced video frames (IMAGE batch).

### FAL AI API Omni Pro Node

This node requires you to know the details of the target fal.ai endpoint.

1.  Add the `FAL AI API Omni Pro Node`.
2.  **`endpoint_id`**: Paste the exact endpoint ID string from fal.ai documentation.
3.  **`api_key`**: Enter your `key_id:key_secret`.
4.  **`parameters_json`**:
    *   Provide a valid JSON object containing **only the non-media parameters** needed by the endpoint (e.g., `prompt`, `seed`, `steps`, booleans, etc.). Check the endpoint's documentation on fal.ai.
    *   **Do not** include keys for media (`image_url`, `video_url`, `audio_url`, `end_image_url`) if connecting the corresponding sockets below.
    *   Example: `{"prompt": "A cat", "seed": 123, "num_steps": 25}`
5.  **Media Inputs (Optional)**: Connect `start_image`, `end_image`, `input_video`, or `input_audio` as needed by the endpoint. The node automatically uploads these and injects the URLs using standard keys (`image_url`, `end_image_url`, `video_url`, `audio_url`). If you manually include one of these keys in the JSON, the uploaded file URL will overwrite it.
6.  **Other Inputs**: Adjust `cleanup_temp_files` and `output_video_fps` if needed.
7.  Output: Resulting image or video frames (IMAGE batch).

## Dependencies

*   `fal-client`
*   `requests`
*   `numpy`
*   `Pillow`
*   `opencv-python`
*   `scipy`

(A `requirements.txt` file is included for easy installation via `pip install -r requirements.txt`.)

## Notes & Troubleshooting

*   **API Key Format:** Ensure the key is `key_id:key_secret`.
*   **Endpoint ID/Parameters:** For the Omni node, always check the official fal.ai documentation for the correct endpoint ID and its specific parameter requirements.
*   **JSON Validity:** Use a validator to check your JSON in the Omni node's `parameters_json` field if you encounter issues.
*   **Temporary Files:** Ensure write permissions for ComfyUI's temp directory. Disable `cleanup_temp_files` only for debugging.

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.
