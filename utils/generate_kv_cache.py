import os
import time
import torch
import pickle
import json

from transformers import AutoModelForCausalLM, AutoTokenizer
from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.storage_backend.serde.cachegen_encoder import CacheGenSerializer
from src.utils import *


if __name__ == "__main__":
    # Get project root directory
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_file_dir)  # parent of utils/
    
    # arguments
    dataset_name = "longchat"
    model_id = "mistral-community/Mistral-7B-v0.2"
    model_name = "mistral7b"
    save_dir = os.path.join(project_dir, "kv_cache", f"{model_name}_{dataset_name}_data")
    encoded_dir = os.path.join(project_dir, "encoded")
    num_gpus = 1
    max_gpu_memory = 48
    start = 0
    end = 1
    
    # Check if save_dir exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
    # Load model and tokenizer
    model, tokenizer = define_model_and_tokenizer(model_id, num_gpus=num_gpus, max_gpu_memory=max_gpu_memory)
    print("Model and tokenizer loaded")
    
    # Load data
    data = load_testcases(DATASET_TO_PATH[dataset_name])
    
    for doc_id in range(start, end):
        print("Saving KV cache for doc: ", doc_id)
        text = data[doc_id]['prompt']
        input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
        st = time.monotonic()
        
        generated = model.generate(input_ids, max_new_tokens = 1, return_dict_in_generate=True)
        torch.cuda.synchronize()
        # print( f"TTFT: {time.monotonic() - st}" )
        
        # Extract key-value cache
        kv = generated['past_key_values']
        kv = list(kv)
        for i in range(len(kv)):
            kv[i] = list(kv[i])
            kv[i][0] = kv[i][0][:, :, :-1][0]
            kv[i][1] = kv[i][1][:, :, :-1][0]
            kv[i] = tuple(kv[i])
        kv = tuple(kv)
        kv_tensor = to_blob(kv)
        
        torch.save(kv_tensor, f"{save_dir}/raw_kv_{doc_id}.pt")
        if doc_id == 0:
            pickle.dump(kv, open(f"{save_dir}/raw_kv_{doc_id}.pkl", "wb"))