import os
import pickle
import argparse
import time

# for socket communication
import socket
import struct
import queue
import pickle

import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue, Barrier
from transformers import AutoTokenizer

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
        # Synchronize with Process_kv_cache for testing
        if layer == 0:
            barrier.wait()

        # Check if termination signal is received
        if not stop_signal.empty() and stop_signal.get() == "STOP":
            print("Process_inference: received stop signal. Terminating.")
            break

        # Get the KV cache from the queue
        kv_cache = kv_queue.get(timeout=20)  # Timeout ensures graceful termination
        kv_cache = kv_cache.cuda()
        decoded_kv = tensor_to_tuple(kv_cache, layer_to_device_id)

        # Run the model
        with torch.no_grad():
            if layer == 0:
                outputs = model.forward_1(input_ids=next_token_input_ids, past_key_values=decoded_kv)
            outputs = model.forward_2(idx=layer, previous_outputs=outputs, past_key_values_layer=decoded_kv)

    # Run the model for the last 2 layers
    with torch.no_grad():
        outputs = model.forward_3(previous_outputs=outputs)
        outputs = model.forward_4(previous_outputs=outputs)
    logits = outputs.logits
    
    print("Process_inference: Inference complete time:", time.time())


# recv_all: Receives all bytes from the socket
def recv_all(sock, length: int) -> bytes:
    data = b''
    while len(data) < length:
        chunk = sock.recv(length - len(data))
        if not chunk:
            raise RuntimeError("Socket connection broken")
        data += chunk
    return data


# Process_kv_cache: Reads KV cache files from disk and sends them to Process_inference
def process_kv_cache(kv_queue, stop_signal, cache_dir, doc_id, barrier):
    # print("[Process_kv_cache] (KV Cache Reader) started")

    # configuration TODO: change this to actual setting
    os.environ["QUANT_LEVEL"] = "2"
    
    # prepare socket for communication
    SERVER_HOST = '143.248.188.5'  # Server IP address
    SERVER_PORT = 50007             # Server port
    doc_id = 0
    
    # create socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((SERVER_HOST, SERVER_PORT))
    print("[Process_kv_cache] Connected to server.")

    for layer in range(32):
        # Synchronize with Process_inference for testing
        if layer == 0:
            barrier.wait()
            print("[Process_kv_cache] start time:", time.time())
            
            # Send the document ID to the server
            s.sendall(struct.pack('>Q', doc_id))
            layer_count_data = recv_all(s, 8)
            layer_count = struct.unpack('>Q', layer_count_data)[0]
        
        if not stop_signal.empty() and stop_signal.get() == "STOP":
            print("[Process_kv_cache] received stop signal. Terminating.")
            break
        
        # Read KV cache file from the socket
        fname_len_data = recv_all(s, 8)
        fname_len = struct.unpack('>Q', fname_len_data)[0]

        fname_data = recv_all(s, fname_len)
        fname = fname_data.decode('utf-8')

        fcontent_len_data = recv_all(s, 8)
        fcontent_len = struct.unpack('>Q', fcontent_len_data)[0]

        fcontent = recv_all(s, fcontent_len)
        encoded_bytes = pickle.loads(fcontent)
        
        # process the KV cache
        lmcache_config = LMCacheEngineConfig.from_defaults(chunk_size=8902)
        meta_data = LMCacheEngineMetadata(model_name="mistral-community/Mistral-7B-v0.2", fmt="huggingface", world_size=1, worker_id=0)
        deserializer = CacheGenDeserializer(lmcache_config, meta_data)
        decoded_kv = deserializer.from_bytes(encoded_bytes)
        
        # put the KV cache in the queue
        kv_queue.put(decoded_kv)
    
    # wait until all layers are processed (gpu tensor reference should be maintained)
    while not kv_queue.empty():
        pass
    
    print("[Process_kv_cache] Shutting down.")


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

    # Create directories if they don't exist
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
    process_a_worker = mp.Process(target=process_inference,
                                  args=(kv_queue, stop_signal, model_name, doc_id, args.dataset_name, barrier))
    process_b_worker = mp.Process(target=process_kv_cache, args=(kv_queue, stop_signal, cache_dir, doc_id, barrier))

    process_a_worker.start()
    process_b_worker.start()

    try:
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
