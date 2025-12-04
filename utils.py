import json
import os
import glob
import uuid
import pandas as pd


def read_json(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        data = read_jsonl(file_path)
    return data


def read_jsonl(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
    except Exception as e:
        data = read_json(file_path)
    return data


def write_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def write_jsonl(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        for line in data:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def read_file(file_path):
    if file_path.endswith(".json"):
        data = read_json(file_path)
    elif file_path.endswith(".jsonl"):
        data = read_jsonl(file_path)
    elif file_path.endswith(".csv"):
        data = pd.read_csv(file_path, header=None)
    else:
        data = None
    return data


def get_uuid():
    return str(uuid.uuid4())


def get_all_files(directory, file_type="json"):
    files = glob.glob(os.path.join(directory, "**", f"*.{file_type}"), recursive=True)
    print(f"{file_type}_files: {len(files)}")
    return files


def get_data_json():
    data_info = {"id": "", "dataset_name": "", "goal": "", "target": "", "meta": ""}
    return data_info


def write_line_jsonl(path, data):
    with open(path, "a") as jsonl_file:
        jsonl_file.write(json.dumps(data, ensure_ascii=False) + "\n")
