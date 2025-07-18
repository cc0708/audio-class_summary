# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import time
import sys
import argparse
import openvino_genai as ov_genai
from openvino import get_version

from summarize_prompt import get_teacher_text_from_folder
import os


def main():
    parser = argparse.ArgumentParser(description="Help command")
    parser.add_argument("-m", "--model", type=str, default=r"C:\Users\intel\Documents\smart classroom\test\Qwen2.5\Qwen2.5-7B\TGWP_INT4_SYM_CW", help="Path to model and tokenizers base directory")
    parser.add_argument("-i", "--input", type=str, default="classroom", help="The folder containing JSON files with speaker diarization results.")
    parser.add_argument("-nw", "--num_warmup", type=int, default=1, help="Number of warmup iterations")
    parser.add_argument("-n", "--num_iter", type=int, default=1, help="Number of iterations")
    parser.add_argument("-mt", "--max_new_tokens", type=int, default=1024, help="Maximal number of new tokens")
    parser.add_argument("-d", "--device", type=str, default="GPU", help="Device")
    
    args = parser.parse_args()
    start = time.time()
    # 1. Define system and user input
    instruction = "You are a helpful assistant."
    input_text = get_teacher_text_from_folder(args.input)

    # 2. ChatML format
    # This is a simple chat prompt format for Qwen2.5
    chat_prompt = (
        f"<|im_start|>system\n{instruction}<|im_end|>\n"
        f"<|im_start|>user\n{input_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    prompt = [chat_prompt]
    if len(prompt) == 0:
        raise RuntimeError(f'Prompt is empty!')

    print(f'openvino runtime version: {get_version()}')

    # Perf metrics is stored in DecodedResults. 
    # In order to get DecodedResults instead of a string input should be a list.
    models_path = args.model
    device = args.device
    num_warmup = args.num_warmup
    num_iter = args.num_iter
    
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = args.max_new_tokens

    scheduler_config = ov_genai.SchedulerConfig()
    scheduler_config.enable_prefix_caching = False
    scheduler_config.max_num_batched_tokens = sys.maxsize

    pipe = ov_genai.LLMPipeline(models_path, device, scheduler_config=scheduler_config)
    
    input_data = pipe.get_tokenizer().encode(prompt)
    prompt_token_size = input_data.input_ids.get_shape()[1]
    print(f"Prompt token size: {prompt_token_size}")
    
    for _ in range(num_warmup):
        print("Warming up...")
        pipe.generate(prompt, config)
    
    res = pipe.generate(prompt, config)
    perf_metrics = res.perf_metrics
    for _ in range(num_iter - 1):
        print("Generating...")
        res = pipe.generate(prompt, config)
        perf_metrics += res.perf_metrics
    
    end = time.time()
    
    print("Prompt\n:", prompt)
    print("Answer\n:", res)
    print("************* Performance metrics *************")
    print(f'Overal Latency: {end-start} s')
    print(f"Output token size: {res.perf_metrics.get_num_generated_tokens()}")
    print(f"Load time: {perf_metrics.get_load_time():.2f} ms")
    print(f"Generate time: {perf_metrics.get_generate_duration().mean:.2f} ± {perf_metrics.get_generate_duration().std:.2f} ms")
    print(f"Tokenization time: {perf_metrics.get_tokenization_duration().mean:.2f} ± {perf_metrics.get_tokenization_duration().std:.2f} ms")
    print(f"Detokenization time: {perf_metrics.get_detokenization_duration().mean:.2f} ± {perf_metrics.get_detokenization_duration().std:.2f} ms")
    print(f"TTFT: {perf_metrics.get_ttft().mean:.2f} ± {perf_metrics.get_ttft().std:.2f} ms")
    print(f"TPOT: {perf_metrics.get_tpot().mean:.2f} ± {perf_metrics.get_tpot().std:.2f} ms")
    print(f"Throughput : {perf_metrics.get_throughput().mean:.2f} ± {perf_metrics.get_throughput().std:.2f} tokens/s")

    with open('response_from_ov_qwen.txt', 'w', encoding='utf-8') as output_file:
            output_file.write(str(res))

if __name__ == "__main__":
    main()
