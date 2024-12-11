import os
import time
import torch
import pickle
import argparse
import json

# from src.utils import *

from fastchat.model import load_model
from transformers import AutoModelForCausalLM, AutoTokenizer


# load utils from src/utils.py 
# TODO: Fix error occuring when loading utils from main.py (fatal error: 'cstddef' file not found)
def define_model_and_tokenizer(model_id, num_gpus=1, max_gpu_memory=48):
    """ Define the model and tokenizer
    """
    if model_id == "Yukang/LongAlpaca-70B-16k":
        from_pretrained_kwargs = {
                                'device_map': 'auto', 
                                'max_memory': {0: '45GiB', 
                                               1: '45GiB', 
                                               2: '45GiB', 
                                               3: '45GiB'}, 
                                'revision': 'main'}
        model = AutoModelForCausalLM.from_pretrained(
                model_id,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                load_in_8bit=True,
                **from_pretrained_kwargs,
            )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    else:
        model, tokenizer = load_model(
                model_id,
                device="cuda",
                num_gpus=num_gpus,
                max_gpu_memory=f"{max_gpu_memory}GiB",
                load_8bit=True,
                cpu_offloading=False,
                debug=False,
            )
    
        
    return model, tokenizer

def load_testcases(test_file):
    with open(test_file, 'r') as json_file:
        json_list = list(json_file)

    test_cases = []
    for test_case in json_list:
        test_case = json.loads(test_case)
        test_cases.append(test_case)

    return test_cases

MAX_API_RETRY = 5
REQ_TIME_GAP = 2
DATASET_TO_PATH = {
    "longchat": "test_data/longchat.jsonl",
    "tqa": "test_data/tqa.jsonl",
    "nqa": "test_data/nqa.jsonl"
}

def to_blob(kv_tuples):
    """ Transform a list of tuples of key and value tensors to a single tensor
    """
    return torch.stack([torch.stack(inner_tuple, dim=0).to("cuda:0") for inner_tuple in kv_tuples], dim=0)


p = argparse.ArgumentParser()

# p.add_argument("--model_id", type = str, default = "lmsys/longchat-7b-16k")
p.add_argument("--model_id", type = str, default = "mistral-community/Mistral-7B-v0.2")
p.add_argument("--save_dir", type=str, default = None)
p.add_argument("--num_gpus", type=int, default = 1)
p.add_argument("--max_gpu_memory", type=int, default=48, help="Default max GPU memory in GiB on A40")
p.add_argument("--path_to_context", type=str, help="The directory where the contexts are stored. ")
p.add_argument("--start", type=int, default = 0)
p.add_argument("--end", type=int, default = 1)
p.add_argument("--dataset_name", type=str)
args = p.parse_args()


if __name__ == "__main__":
    # Check if save_dir exists
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
        
    # Load model and tokenizer
    model, tokenizer = define_model_and_tokenizer(args.model_id, num_gpus=args.num_gpus, max_gpu_memory=args.max_gpu_memory)
    print("Model and tokenizer loaded")
    
    # Load data
    data = load_testcases(DATASET_TO_PATH[args.dataset_name])
    for doc_id in range(args.start, args.end):
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

        # TODO: Understand what this code does
        for i in range(len(kv)):
            kv[i] = list(kv[i])
            kv[i][0] = kv[i][0][:, :, :-1][0]
            kv[i][1] = kv[i][1][:, :, :-1][0]
            kv[i] = tuple(kv[i])
        kv = tuple(kv)
        
        # Convert layerwise key-value cache to single tensor
        kv_tensor_list = []
        for i in range(model.config.num_hidden_layers):
            # kv_tensor = to_blob(kv)
            kv_tensor = to_blob((kv[i],))
            kv_tensor_list.append(kv_tensor)
        
        # Save layerwise key-value cache (first file is saved as pkl)
        for i in range(model.config.num_hidden_layers):
            torch.save(kv_tensor_list[i], f"{args.save_dir}/raw_kv_{doc_id}_layer_{i}.pt")
            if doc_id == 0:
                pickle.dump(kv, open(f"{args.save_dir}/raw_kv_{doc_id}_layer_{i}.pkl", "wb"))