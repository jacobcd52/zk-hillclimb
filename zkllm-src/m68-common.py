# Shared config for running zkLLM natively on JackFram/llama-68m (MHA LLaMA arch).
# Mirrors the hardcoded Llama-2 plumbing in the original llama-*.py scripts.
MODEL_CARD = "JackFram/llama-68m"
WORKDIR = "./zkllm-workdir/llama-68m"
CACHE_DIR = "./model-storage"
LOG_SCALING_FACTOR = 16
LOG_OFF_FACTOR = 5
