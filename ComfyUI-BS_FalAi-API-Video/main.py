from .nodes import node_i2v, node_t2v, node_omni, node_lipsync

NODE_CLASS_MAPPINGS = {
    "FalAPIVideoGeneratorI2V": node_i2v.FalAPIVideoGeneratorI2V,
    "FalAPIVideoGeneratorT2V": node_t2v.FalAPIVideoGeneratorT2V,
    "FalAPIOmniProNode": node_omni.FalAPIOmniProNode,
    "FalAILipSyncNode": node_lipsync.FalAILipSyncNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FalAPIVideoGeneratorI2V": "FAL AI Image-to-Video",
    "FalAPIVideoGeneratorT2V": "FAL AI Text-to-Video",
    "FalAPIOmniProNode": "FAL AI API Omni Pro Node",
    "FalAILipSyncNode": "FAL AI API LipSync Node (v1.9/v2.0)",
}
