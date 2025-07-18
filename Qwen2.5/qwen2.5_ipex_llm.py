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

import torch
import time
import argparse
from summarize_prompt import get_teacher_text_from_folder
from ipex_llm.utils import BenchmarkWrapper
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Summarize text using generate() API for Qwen2.5 model')
    
    parser.add_argument('--input', type=str, default='classroom', 
                        help="The folder containing JSON files with speaker diarization results.")
    parser.add_argument('--repo-id-or-model-path', type=str, default=r"C:\Users\intel\.cache\modelscope\hub\Qwen\Qwen2.5-7B-Instruct",
                        help='The Hugging Face or ModelScope repo id for the Qwen2.5 model to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--prompt', type=str, default="AI是什么？",
                        help='Prompt to infer') 
    parser.add_argument('--n-predict', type=int, default=1024,
                        help='Max tokens to predict')
    parser.add_argument('--modelscope', action="store_true", default=False, 
                        help="Use models from modelscope")

    args = parser.parse_args()
    
    if args.modelscope:
        from modelscope import AutoTokenizer
        model_hub = 'modelscope'
    else:
        from transformers import AutoTokenizer
        model_hub = 'huggingface'
    
    model_path = args.repo_id_or_model_path


    from ipex_llm.transformers import AutoModelForCausalLM
    # Load model in 4 bit,
    # which convert the relevant layers in the model into INT4 format
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 load_in_4bit=True,
                                                 optimize_model=True,
                                                 trust_remote_code=True,
                                                 use_cache=True,
                                                 model_hub=model_hub)
    model = model.half().to("xpu")
    model = BenchmarkWrapper(model, do_print=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              trust_remote_code=True)

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
        # warmup
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=args.n_predict
        )
        
        st = time.time()
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=args.n_predict
        )
        print("inference done")
        torch.xpu.synchronize()
        print("sycnhronize done")
        end = time.time()
        generated_ids = generated_ids.cpu()
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        print("generated_ids done")

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(response)
        print(f'Inference time: {end-st} s')
        # print('-'*20, 'Prompt', '-'*20)
        # print(prompt)
        # print('-'*20, 'Output', '-'*20)
        # 可选：保存到新文件
        with open('response.txt', 'w', encoding='utf-8') as output_file:
            output_file.write(response)