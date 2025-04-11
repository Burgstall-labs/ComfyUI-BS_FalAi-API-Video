
import json

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
            "schema_str": "[Image_url:String], [Prompt:String], [seed:Integer], [negative_prompt:String], [resolution:ResolutionEnum], [duration:Integer]",
        },
        "Pika Image to Video (v2.1) Image to Video": {
            "endpoint": "fal-ai/pika/v2.1/image-to-video",
            "resolutions": ["720p", "1080p"], "aspect_ratios": ["16:9", "9:16", "1:1", "4:5", "5:4", "3:2", "2:3"], "durations": [],
            "schema_str": "[Image_url:String], [Prompt:String], [seed:Integer], [negative_prompt:String], [resolution:ResolutionEnum], [duration:Integer]",
        },
        "Vidu Image to Video": {
            "endpoint": "fal-ai/vidu/image-to-video",
            "resolutions": [], "aspect_ratios": ["16:9", "9:16", "1:1"], "durations": [3],
            "schema_str": "[image_url:String], [prompt:String], [negative_prompt:String], [seed:Integer], [num_frames:Integer], [fps:Integer], [guidance_scale:Float], [num_inference_steps:Integer]",
        },
         "WAN Pro Image to Video": {
            "endpoint": "fal-ai/wan-pro/image-to-video",
            "resolutions": ["480p", "720p"], "aspect_ratios": ["auto", "16:9", "9:16", "1:1"], "durations": [5],
            "schema_str": "[Prompt:String], [negative_prompt:String], [image_url:String], [num_frames:Integer], [frames_per_second:Integer], [seed:Integer], [motion:Integer], [resolution:ResolutionEnum], [num_inference_steps:Integer]",
        },
        "Hunyuan Video (Image to Video)": {
            "endpoint": "fal-ai/hunyuan-video-image-to-video",
            "resolutions": ["256p", "512p"], "aspect_ratios": ["1:1"], "durations": [],
            "schema_str": "[Prompt:String], [image_url:String], [seed:Integer], [negative_prompt:String]",
        },
        "LTX Video v0.95 Image to Video": {
            "endpoint": "fal-ai/ltx-video-v095/image-to-video",
            "resolutions": ["256p", "512p"], "aspect_ratios": ["16:9", "9:16", "1:1"], "durations": [2,3,4,5,6,7,8,9,10],
            "schema_str": "[Prompt:String], [image_url:String], [negative_prompt:String], [seed:Integer], [num_frames:Integer], [fps:Integer], [guidance_scale:Float], [num_inference_steps:Integer]",
        },
          "Luma Dream Machine (Image to Video) Image to Video": {
            "endpoint": "fal-ai/luma-dream-machine/image-to-video",
            "resolutions": ["540p", "720p", "1080p"], "aspect_ratios": ["16:9", "9:16", "4:3", "3:4", "21:9", "9:21"], "durations": [5],
            "schema_str": "[Prompt:String], [image_url:String], [end_image_url:String], [aspect_ratio:AspectRatioEnum], [loop:Boolean], [resolution:ResolutionEnum], [duration:DurationEnum]",
        },
        "Luma Ray 2 (Image to Video) Image to Video": {
            "endpoint": "fal-ai/luma-dream-machine/ray-2/image-to-video",
            "resolutions": ["540p", "720p", "1080p"], "aspect_ratios": ["16:9", "9:16", "4:3", "3:4", "21:9", "9:21"], "durations": [5],
            "schema_str": "[Prompt:String], [image_url:String], [end_image_url:String], [aspect_ratio:AspectRatioEnum], [loop:Boolean], [resolution:ResolutionEnum], [duration:DurationEnum]",
        },
        "Hunyuan Video (Image to Video - LoRA)": {
            "endpoint": "fal-ai/hunyuan-video-img2vid-lora",
            "resolutions": ["256p", "512p"], "aspect_ratios": ["1:1"], "durations": [],
            "schema_str": "[Prompt:String], [image_url:String], [seed:Integer], [negative_prompt:String]",
        },
        "PixVerse v3.5: Image to Video Image to Video": {
            "endpoint": "fal-ai/pixverse/v3.5/image-to-video",
            "resolutions": ["360p", "540p", "720p", "1080p"], "aspect_ratios": ["16:9", "4:3", "1:1", "3:4", "9:16"], "durations": [5, 8],
            "schema_str": "[Prompt:String], [aspect_ratio:AspectRatioEnum], [resolution:ResolutionEnum], [duration:DurationEnum], [negative_prompt:String], [style:Enum], [seed:Integer], [image_url:String]",
        },
        "PixVerse v3.5: Image to Video Fast Image to Video": {
            "endpoint": "fal-ai/pixverse/v3.5/image-to-video/fast",
            "resolutions": ["360p", "540p", "720p"], "aspect_ratios": ["16:9", "4:3", "1:1", "3:4", "9:16"], "durations": [],
            "schema_str": "[Prompt:String], [aspect_ratio:AspectRatioEnum], [resolution:ResolutionEnum], [negative_prompt:String], [style:Enum], [seed:Integer], [image_url:String]",
        },
         "LTX Video Image to Video": {
            "endpoint": "fal-ai/ltx-video/image-to-video",
            "resolutions": ["256p", "512p"], "aspect_ratios": ["16:9", "9:16", "1:1"], "durations": [2,3,4,5,6,7,8,9,10],
            "schema_str": "[Prompt:String], [image_url:String], [negative_prompt:String], [seed:Integer], [num_frames:Integer], [fps:Integer], [guidance_scale:Float], [num_inference_steps:Integer]",
        },
        "CogVideoX 5B Image to Video": {
            "endpoint": "fal-ai/cogvideox-5b/image-to-video",
            "resolutions": [], "aspect_ratios": ["16:9", "9:16", "1:1"], "durations": [2,3,4,5,6,7,8,9,10],
            "schema_str": "[Prompt:String], [image_url:String], [seed:Integer], [negative_prompt:String], [num_frames:Integer], [fps:Integer], [guidance_scale:Float], [num_inference_steps:Integer]",
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
            "schema_str": "[prompt:String], [negative_prompt:String], [seed:Integer], [resolution:ResolutionEnum], [duration:Integer]",
        },
        "Luma Dream Machine Text to Video": {
            "endpoint": "fal-ai/luma-dream-machine/text-to-video",
            "resolutions": ["540p", "720p", "1080p"], "aspect_ratios": ["16:9", "9:16", "4:3", "3:4", "21:9", "9:21"], "durations": [5],
            "schema_str": "[prompt:String], [seed:Integer], [aspect_ratio:AspectRatioEnum], [loop:Boolean], [resolution:ResolutionEnum], [duration:DurationEnum]",
        },
        "PixVerse v4 Text to Video": {
            "endpoint": "fal-ai/pixverse/v4/text-to-video",
            "resolutions": ["360p", "540p", "720p", "1080p"], "aspect_ratios": ["16:9", "4:3", "1:1", "3:4", "9:16"], "durations": [5, 8],
            "schema_str": "[prompt:String], [negative_prompt:String], [style:Enum], [seed:Integer], [aspect_ratio:AspectRatioEnum], [resolution:ResolutionEnum], [duration:DurationEnum]",
        },
        "MiniMax (Hailuo AI) Video 01 Text to Video": {
            "endpoint": "fal-ai/minimax/video-01/text-to-video",
            "resolutions": [], "aspect_ratios": [], "durations": [],
            "schema_str": "[prompt:String]",
        },
        "Hunyuan Video Text to Video": {
            "endpoint": "fal-ai/hunyuan-video",
            "resolutions": ["256p", "512p"], "aspect_ratios": ["1:1"], "durations": [],
            "schema_str": "[prompt:String], [seed:Integer], [negative_prompt:String]",
        },
    },
}

ALL_MODEL_NAMES_I2V = sorted(list(MODEL_CONFIGS["image_to_video"].keys()))
ALL_MODEL_NAMES_T2V = sorted(list(MODEL_CONFIGS["text_to_video"].keys()))
ALL_RESOLUTIONS = sorted(list(set(res for cat in MODEL_CONFIGS.values() for cfg in cat.values() for res in cfg["resolutions"] if res)))
ALL_ASPECT_RATIOS = sorted(list(set(ar for cat in MODEL_CONFIGS.values() for cfg in cat.values() for ar in cfg["aspect_ratios"] if ar)))
if not ALL_RESOLUTIONS: ALL_RESOLUTIONS = ["720p", "1080p", "512p", "576p"]
if not ALL_ASPECT_RATIOS: ALL_ASPECT_RATIOS = ["16:9", "9:16", "1:1", "4:3", "3:4"]
if "auto" not in ALL_ASPECT_RATIOS: ALL_ASPECT_RATIOS.insert(0, "auto")
ALL_RESOLUTIONS.insert(0, "auto")

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

for category, models in MODEL_CONFIGS.items():
    for name, config in models.items():
        config['expected_params'] = parse_schema(config['schema_str'])
