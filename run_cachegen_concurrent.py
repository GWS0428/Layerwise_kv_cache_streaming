import os
import pickle
import argparse

import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue, Barrier
from transformers import AutoTokenizer
from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.storage_backend.serde.cachegen_encoder import CacheGenSerializer
from lmcache.storage_backend.serde.cachegen_decoder import CacheGenDeserializer

from src.utils import *
from model.mistral_custom import MistralForCausalLM_custom


# Process_inference : Runs LLM inference layer-wise on gpu
def process_inference(kv_queue, stop_signal, model_name, doc_id, dataset_name, barrier):
    print("Process_inference (LLM Inference) started for doc_id:", doc_id)
    
    # test setting TODO: change this to actual setting
    device = torch.device("cuda:0")
    layer_to_device_id = {}
    device_index = 0
    num_layers = 32
    for l in range(num_layers):
        layer_to_device_id[l] = device_index

    # dataset loading
    data = load_testcases(DATASET_TO_PATH[dataset_name])
    os.environ["QUANT_LEVEL"] = "2"
    
    # Load the model on GPU
    model = MistralForCausalLM_custom.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()
    print("Process_inference: Model loaded successfully.")
    
    # Load the prompt
    text = data[doc_id]['prompt']
    input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
    next_token_input_ids = input_ids.clone().cuda()  # Clone input_ids to avoid mutation        

    # run the model with the decoded_kv in layerwise manner
    outputs = None
    for layer in range(num_layers):
        # Synchronize with Process B before entering the loop
        barrier.wait()
        
        # Check if termination signal is received
        if not stop_signal.empty() and stop_signal.get() == "STOP":
            print("Process_inference: received stop signal. Terminating.")
            break
        # try:
        kv_cache = kv_queue.get(timeout=20)  # Timeout ensures graceful termination
        print(f"Process_inference: Received KV cache for Layer {layer}")
        
        kv_cache = kv_cache.cuda()
        decoded_kv = tensor_to_tuple(kv_cache, layer_to_device_id)
        
        if layer == 0:
            outputs = model.forward_1(input_ids=next_token_input_ids, past_key_values=decoded_kv)
        outputs = model.forward_2(idx=layer, previous_outputs=outputs, past_key_values_layer=decoded_kv)
        print(f"Layer {layer} done")
        # except Exception as e:
        #     print(f"Process_inference: Exception occurred: {e}")
    outputs = model.forward_3(previous_outputs=outputs)
    outputs = model.forward_4(previous_outputs=outputs)
    logits = outputs.logits 


# Process_kv_cache: Reads KV cache files from disk and sends them to Process_inference
def process_kv_cache(kv_queue, stop_signal, cache_dir, doc_id, barrier):
    print("Process_kv_cache (KV Cache Reader) started")
    
    # configuration TODO: change this to actual setting
    os.environ["QUANT_LEVEL"] = "2"
    
    for layer in range(32):
        # Synchronize with Process B before entering the loop
        barrier.wait()
        
        if not stop_signal.empty() and stop_signal.get() == "STOP":
            print("Process B received stop signal. Terminating.")
            break
        
        with open(f"{cache_dir}/{doc_id}_layer_{layer}.pkl", "rb") as f:
            encoded_bytes = pickle.load(f)
        lmcache_config = LMCacheEngineConfig.from_defaults(chunk_size=8902)
        meta_data = LMCacheEngineMetadata(model_name="mistral-community/Mistral-7B-v0.2", fmt="huggingface", world_size=1, worker_id=0)
        deserializer = CacheGenDeserializer(lmcache_config, meta_data)
        decoded_kv = deserializer.from_bytes(encoded_bytes)

        print(f"Process_kv_cache: Sending KV cache for Layer {layer}")
        kv_queue.put(decoded_kv)


if __name__ == "__main__":
    # Use spawn method for GPU memory sharing
    mp.set_start_method('spawn', force=True)

    p = argparse.ArgumentParser()

    p.add_argument("--model_id", type=str, default="lmsys/longchat-7b-16k")
    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument("--num_gpus", type=int, default=1)
    p.add_argument("--max_gpu_memory", type=int, default=48, help="Default max GPU memory in GiB on A40")
    p.add_argument("--path_to_context", type=str, help="The directory where the contexts are stored.")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=1)
    p.add_argument("--encoded_dir", type=str, default=None)
    p.add_argument("--results_dir", type=str, default=None)
    p.add_argument("--results_str", type=str, default="results")
    p.add_argument("--dataset_name", type=str)
    p.add_argument("--calculate_metric", type=int)

    args = p.parse_args()

    if not os.path.exists(args.encoded_dir):
        os.makedirs(args.encoded_dir, exist_ok=True)
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir, exist_ok=True)

    # Configuration
    model_name = "mistral-community/Mistral-7B-v0.2"  # HuggingFace model name
    cache_dir = args.encoded_dir  # Directory where KV cache files are stored
    doc_id = 0  # Document ID
    
    # Shared Queue for KV cache communication
    kv_queue = mp.Queue(maxsize=40)
    stop_signal = mp.Queue()  # To signal termination to both processes
    barrier = Barrier(2)  # Barrier for synchronizing 2 processes

    # Start Process A and B
    process_a_worker = mp.Process(target=process_inference, args=(kv_queue, stop_signal, model_name, doc_id, args.dataset_name, barrier))
    process_b_worker = mp.Process(target=process_kv_cache, args=(kv_queue, stop_signal, cache_dir, doc_id, barrier))

    process_a_worker.start()
    process_b_worker.start()

    try:
        # Run for a fixed duration or until user interrupts
        time.sleep(100)  # Run for 100 seconds
    except KeyboardInterrupt:
        print("Terminating processes...")
    finally:
        # Send stop signals
        stop_signal.put("STOP")
        stop_signal.put("STOP")
        
        # Wait for processes to terminate
        process_a_worker.join()
        process_b_worker.join()

    print("Both processes have terminated. Exiting main.")
