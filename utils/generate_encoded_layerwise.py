
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import numpy as np
import os
import time
import pickle
import torch
from src.attention_monkey_patch import replace_llama_forward_with_reuse_forward
from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.storage_backend.serde.cachegen_encoder import CacheGenSerializer
from lmcache.storage_backend.serde.cachegen_decoder import CacheGenDeserializer
import json
from src.utils import *


if __name__ == "__main__":
    # arguments
    doc_id = 0
    num_layers = 32
    dataset_name = "longchat"
    model_id = "mistral-community/Mistral-7B-v0.2"
    model_name = "mistral7b"
    save_dir = f"./{model_name}_{dataset_name}_data"
    encoded_dir = "./encoded"
    
    # Check if encoded_dir and result_dir is exists
    if not os.path.exists(encoded_dir):
        os.makedirs(encoded_dir, exist_ok=True)
        
    # Read data from jsonl
    avg_size = []
    kv_tokens = []
    layer_to_device_id = {}
    os.environ["QUANT_LEVEL"] = "2"
    data = load_testcases(DATASET_TO_PATH[dataset_name])
    
    # Start encoding in layerwise manner
    kv = pickle.load(open(f"{save_dir}/raw_kv_{doc_id}.pkl", "rb"))
    
    for i in range(len(kv)):
        layer_to_device_id[i] = kv[i][0].device.index
    
    for i in range(num_layers):
        key_value = torch.load(f"{save_dir}/raw_kv_{doc_id}_layer_{i}.pt")
        lmcache_config = LMCacheEngineConfig.from_defaults(chunk_size=key_value.shape[-2])
        meta_data = LMCacheEngineMetadata(model_name=model_id, fmt="huggingface", world_size=1, worker_id=0)
        cachegen_serializer = CacheGenSerializer(lmcache_config, meta_data)
        bytes = cachegen_serializer.to_bytes(key_value)
        pickle.dump(bytes, open(f"{encoded_dir}/{doc_id}_layer_{i}.pkl", "wb"))
