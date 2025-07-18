#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from transformers import AutoTokenizer
import intel_extension_for_pytorch as ipex
from neural_compressor.transformers import AutoModelForCausalLM, RtnConfig
import torch
import time
import argparse
from summarize_prompt import get_teacher_text_from_folder

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Summarize text using generate() API for Qwen2.5 model')
    
    parser.add_argument('--input', type=str, default='classroom', 
                        help="The folder containing JSON files with speaker diarization results.")
    parser.add_argument('--repo-id-or-model-path', type=str, default=r"C:\Users\intel\.cache\modelscope\hub\Qwen\Qwen2.5-7B-Instruct",
                        help='The Hugging Face or ModelScope repo id for the Qwen2.5 model to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--n-predict', type=int, default=1024,
                        help='Max tokens to predict')

    args = parser.parse_args()
    
    model_path = args.repo_id_or_model_path
    woq_checkpoint_path = "./Qwen2.5-7B_woq_int4"

    start = time.time()
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              trust_remote_code=True)
    
    
    if os.path.exists(woq_checkpoint_path):
        print("Directly loading already quantized model")
    # Load the already quantized model
        model = AutoModelForCausalLM.from_pretrained(
            woq_checkpoint_path, 
            trust_remote_code=True, 
            device_map="xpu", 
            torch_dtype=torch.float16
        )
        model = model.to(memory_format=torch.channels_last)
        woq_quantization_config = getattr(model, "quantization_config", None)
    else:
    # Define the quantization configuration
        woq_quantization_config = RtnConfig(
            compute_dtype="fp16", 
            weight_dtype="int4_fullrange", 
            scale_dtype="fp16", 
            group_size=64
        )

        model = AutoModelForCausalLM.from_pretrained(model_path,
                                                    device_map="xpu",
                                                    quantization_config=woq_quantization_config,
                                                    trust_remote_code=True)
        # Save the quantized model and tokenizer
        model.save_pretrained(woq_checkpoint_path)
        tokenizer.save_pretrained(woq_checkpoint_path)

    # Set the model to evaluation mode and move it to the XPU
    model = model.eval().to("xpu")
    model = model.to(memory_format=torch.channels_last)

    # Optimize the model with Intel Extension for PyTorch (IPEX)
    model = ipex.llm.optimize(
        model.eval(), 
        device="xpu", 
        inplace=True, 
        quantization_config=woq_quantization_config
)

    prompt = get_teacher_text_from_folder(args.input)

    # Generate predicted tokens
    with torch.inference_mode():
        # The following code for generation is adapted from https://huggingface.co/Qwen/Qwen2.5-7B-Instruct#quickstart
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to("xpu")
        # # warmup
        print("warmup...")
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=args.n_predict
        )

        print("generate...")
        generate_start = time.time()
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=args.n_predict
        )
        print("inference done")
        torch.xpu.synchronize()
        print("sycnhronize done")
        generate_end = time.time()
        generated_ids = generated_ids.cpu()
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        print("generated_ids done")

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(response)
        end = time.time()
        print(f'Inference time: {generate_end-generate_start} s')
        print(f'Overal Latency: {end-start} s')
        # print('-'*20, 'Prompt', '-'*20)
        # print(prompt)
        # print('-'*20, 'Output', '-'*20)
        
        # 可选：保存到新文件
        with open('response_from_ipex_qwen.txt', 'w', encoding='utf-8') as output_file:
            output_file.write(response)