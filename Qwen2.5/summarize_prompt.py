import yaml
import json
import os
import re

def get_prompt_from_folder(folder_path):
    # 读取 prompt.yml 文件
    with open('prompt.yml', 'r', encoding='utf-8') as yml_file:
        prompt_template = yaml.safe_load(yml_file)

    # 获取所有以matched_results_part_开头的json文件，并按数字排序
    json_files = []
    pattern = re.compile(r"matched_results_part_(\d+)\.json")
    for f in os.listdir(folder_path):
        m = pattern.match(f)
        if m:
            json_files.append((int(m.group(1)), f))
    json_files.sort()  # 按数字从小到大排序

    # 拼接所有 json 文件的数据
    all_json_data = []
    for _, json_file in json_files:
        json_path = os.path.join(folder_path, json_file)
        print("open json file ", json_path)
        with open(json_path, 'r', encoding='utf-8') as jf:
            json_data = json.load(jf)
            all_json_data.append(json_data)

    # 将所有 JSON 数据格式化为字符串并拼接
    json_str = ""
    for jd in all_json_data:
        json_str += json.dumps(jd, ensure_ascii=False, indent=2) + "\n"

    # 替换模板中的占位符
    final_prompt = prompt_template['prompt'].replace('{在此粘贴 JSON 数据}', json_str.strip())

    # 可选：保存到新文件
    with open('final_prompt.txt', 'w', encoding='utf-8') as output_file:
        output_file.write(final_prompt)
    return final_prompt

def get_teacher_text_from_folder(folder):

    json_files = sorted([f for f in os.listdir(folder) if f.startswith('matched_results_part_') and f.endswith('.json')])

    all_teacher_text = ""

    for filename in json_files:
        filepath = os.path.join(folder, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Compatible with common structures
            if isinstance(data, dict) and "results" in data:
                items = data["results"]
            elif isinstance(data, list):
                if data and isinstance(data[0], dict) and "results" in data[0]:
                    items = []
                    for entry in data:
                        items.extend(entry["results"])
                else:
                    items = data
            else:
                continue

            for item in items:
                if item.get("speaker") == "teacher":
                    all_teacher_text += item.get("text", "")

    with open('prompt.yml', 'r', encoding='utf-8') as yml_file:
        prompt_template = yaml.safe_load(yml_file)

    final_prompt = prompt_template['prompt'].replace('{在此粘贴 JSON 数据}', all_teacher_text)

    return final_prompt

if __name__ == "__main__":
    get_prompt_from_folder()