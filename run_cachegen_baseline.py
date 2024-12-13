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
from transformers import AutoTokenizer

from src.utils import *
from model.mistral_custom import MistralForCausalLM_custom


# Process_inference : Runs baseline LLM inference on gpu
def process_inference(kv_queue, model, next_token_input_ids, layer_to_device_id, kv_cache=None):

    # run the model with the decoded_kv
    outputs = None
    
    if kv_cache is None:
        decoded_kv = kv_queue.get(timeout=20)  # Timeout ensures graceful termination
    else:
        decoded_kv = kv_cache

    decoded_kv = decoded_kv.cuda()
    decoded_kv = tensor_to_tuple(decoded_kv, layer_to_device_id)

    with torch.no_grad():
        outputs = model.forward(input_ids=next_token_input_ids, past_key_values=decoded_kv)
    logits = outputs.logits
    
    llm_end = time.time()


# recv_all: Receives all bytes from the socket
def recv_all(sock, length: int) -> bytes:
    data = b''
    while len(data) < length:
        chunk = sock.recv(length - len(data))
        if not chunk:
            raise RuntimeError("Socket connection broken")
        data += chunk
    return data


def create_socket(SERVER_HOST, SERVER_PORT):
    """Create and return a connected socket."""
    
    # create socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
    s.connect((SERVER_HOST, SERVER_PORT))
    print("[create_socket] Connected to server.")
            
    return s


def close_socket(sock):
    """Close the socket connection."""
    sock.close()


# Process_kv_cache: Reads KV cache files from socket and sends them to Process_inference
def process_kv_cache(kv_queue, s, doc_id):
    # Send the document ID to the server
    s.sendall(struct.pack('>Q', doc_id))
    layer_count_data = recv_all(s, 8)
    layer_count = struct.unpack('>Q', layer_count_data)[0]

    # Read KV cache file from the socket
    fname_len_data = recv_all(s, 8)
    fname_len = struct.unpack('>Q', fname_len_data)[0]

    fname_data = recv_all(s, fname_len)
    fname = fname_data.decode('utf-8')

    fcontent_len_data = recv_all(s, 8)
    fcontent_len = struct.unpack('>Q', fcontent_len_data)[0]

    fcontent = recv_all(s, fcontent_len)
    encoded_bytes = pickle.loads(fcontent)
    
    # Deserialize the KV cache
    lmcache_config = LMCacheEngineConfig.from_defaults(chunk_size=8902)
    meta_data = LMCacheEngineMetadata(model_name="mistral-community/Mistral-7B-v0.2", fmt="huggingface", world_size=1, worker_id=0)
    deserializer = CacheGenDeserializer(lmcache_config, meta_data)
    decoded_kv = deserializer.from_bytes(encoded_bytes)

    # Put the KV cache in the queue
    kv_queue.put(decoded_kv)
    
    # Close the socket
    close_socket(s)
    

if __name__ == "__main__":
    # Use spawn method for GPU memory sharing
    mp.set_start_method('spawn', force=True)

    p = argparse.ArgumentParser()

    # p.add_argument("--model_id", type=str, default="lmsys/longchat-7b-16k")
    p.add_argument("--model_id", type=str, default="mistral-community/Mistral-7B-v0.2")
    p.add_argument("--save_dir", type=str, default=".")
    p.add_argument("--num_gpus", type=int, default=1)
    p.add_argument("--max_gpu_memory", type=int, default=48, help="Default max GPU memory in GiB on A40")
    p.add_argument("--path_to_context", type=str, help="The directory where the contexts are stored.")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=1)
    p.add_argument("--encoded_dir", type=str, default=None)
    p.add_argument("--results_dir", type=str, default=None)
    p.add_argument("--results_str", type=str, default="results")
    p.add_argument("--dataset_name", type=str, default="longchat")
    p.add_argument("--calculate_metric", type=int)

    args = p.parse_args()

    # Create directories if they don't exist
    if not os.path.exists(args.encoded_dir):
        os.makedirs(args.encoded_dir, exist_ok=True)
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir, exist_ok=True)

    # Configuration for model inference
    model_name = "mistral-community/Mistral-7B-v0.2"  # HuggingFace model name
    cache_dir = args.encoded_dir  # Directory where KV cache files are stored
    doc_id = 0  # Document ID
    dataset_name = "longchat"  # Dataset name
    
    # Constants for socket communication
    SERVER_HOST = '143.248.188.5'  # Server IP address
    SERVER_PORT = 50007             # Server port
    doc_id = 0

    # Shared Queue for KV cache communication
    kv_queue = queue.Queue()
    
    # setups for model inference
    num_layers = 32
    device_index = 0 # TODO: change this to actual setting
    device = torch.device("cuda:0")
    layer_to_device_id = {i: device_index for i in range(32)}

    # dataset loading
    os.environ["QUANT_LEVEL"] = "2"
    data = load_testcases(DATASET_TO_PATH[dataset_name])

    # Load the model on GPU
    model = MistralForCausalLM_custom.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()

    # Load the prompt
    text = data[doc_id]['prompt']
    input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
    next_token_input_ids = input_ids.clone().cuda()  # Clone input_ids to avoid mutation

    # Create a socket connection
    s = create_socket(SERVER_HOST, SERVER_PORT)

    # Start Process_inference and process_kv_cache 
    start_time = time.time()
    process_kv_cache(kv_queue, s, doc_id)
    process_inference(kv_queue, model, next_token_input_ids, layer_to_device_id)
    end_time = time.time()
    print(f"Time: {end_time - start_time} seconds")

    print("Exiting main.")
