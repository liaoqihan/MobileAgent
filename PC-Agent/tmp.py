import os
from modelscope import snapshot_download, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

local_cache_dir = "./model_cache"
model_dir = os.path.join(local_cache_dir, "Qwen-VL-Chat-Int4")
if not os.path.exists(model_dir):
    print("Downloading Qwen-VL-Chat-Int4 model...")
    model_dir = snapshot_download("qwen/Qwen-VL-Chat-Int4", revision='v1.0.0', cache_dir=local_cache_dir)



groundingdino_dir = os.path.join(local_cache_dir, "AI-ModelScope")
if not os.path.exists(groundingdino_dir):
    print("Downloading AI-ModelScope/GroundingDINO model...")
    groundingdino_dir = snapshot_download('AI-ModelScope/GroundingDINO', revision='v1.0.0', cache_dir=local_cache_dir)
