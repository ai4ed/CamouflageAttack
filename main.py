import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml
from utils import read_file, write_line_jsonl
from datetime import datetime
import os
import multiprocessing as mp
import logging
from agents import run, StrategyConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    args = parser.parse_args()
    return args


def parse_config(config_path):
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    return config


def build_data(prompt):
    message = [{"role": "user", "content": prompt}]
    return message


def get_logger(config, time_str):
    log_file_dir = config["log"]["dir"]
    os.makedirs(log_file_dir, exist_ok=True)
    log_file_path = os.path.join(log_file_dir, f"{config['task_name']}-{time_str}.log")
    logger = logging.getLogger("Experimental Logger")
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    return logger


def attack_worker(data, device, config, queue, process_id):
    strategy_config = StrategyConfig(**config["strategy"])
    max_new_tokens = config["model"]["max_new_tokens"]

    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["path"], torch_dtype=config["model"]["dtype"]
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for line in data:
        message_original = build_data(line["goal"])

        result = run(model, tokenizer, message_original, line["target"], strategy_config)
        message_attack = build_data(
            message_original[-1]["content"] + " " + result.best_string
        )
        input_attack = tokenizer.apply_chat_template(
            message_attack, add_generation_prompt=True, return_tensors="pt"
        ).to(device)
        output_attack = model.generate(
            input_attack, do_sample=False, max_new_tokens=max_new_tokens
        )
        response_attack = tokenizer.batch_decode(
            output_attack[:, input_attack.shape[1] :], skip_special_tokens=True
        )[0]
        attack_string = result.best_string
        best_loss = result.best_loss

        input_original = tokenizer.apply_chat_template(
            message_original, add_generation_prompt=True, return_tensors="pt"
        ).to(device)
        output_original = model.generate(
            input_original, do_sample=False, max_new_tokens=max_new_tokens
        )
        response_original = tokenizer.batch_decode(
            output_original[:, input_original.shape[1] :], skip_special_tokens=True
        )[0]

        line["attack_string"] = attack_string
        line["message_original"] = message_original[-1]["content"]
        line["message_attack"] = message_attack[-1]["content"]
        line["response_original"] = response_original
        line["response_attack"] = response_attack
        line["best_loss"] = best_loss
        line["process_id"] = process_id

        queue.put(line)

    queue.put(process_id)


def main():
    args = parse_args()
    config_path = args.config_path
    config = parse_config(config_path)
    devices = config["model"]["devices"]
    num_devices = len(devices)

    time_str = datetime.now().strftime("%Y%m%d%H%M%S")
    logger = get_logger(config, time_str)
    data_save_dir = os.path.join(config["result"]["dir"])
    os.makedirs(data_save_dir, exist_ok=True)
    data_save_file = os.path.join(
        data_save_dir,
        config["task_name"] + "-" + time_str + ".jsonl",
    )

    data = read_file(config["data"]["path"])
    data_length = len(data)
    data_chunk_size = data_length // num_devices
    data_chunk_remainder = data_length % num_devices
    data_chunks = [data_chunk_size] * num_devices
    while data_chunk_remainder > 0:
        data_chunks[data_chunk_remainder] += 1
        data_chunk_remainder -= 1

    results = []
    processes = []
    finished = [False] * num_devices
    mp.set_start_method("spawn")
    queue = mp.Queue()

    for i, device in enumerate(devices):
        data_start_idx = sum(data_chunks[:i])
        data_end_idx = data_start_idx + data_chunks[i]
        p = mp.Process(
            target=attack_worker,
            args=(data[data_start_idx:data_end_idx], device, config, queue, i),
        )
        processes.append(p)
        p.start()

    try:
        while len(results) < data_length:
            for i, p in enumerate(processes):
                if not p.is_alive() and not finished[i]:
                    raise Exception(f"Subprocess {i} terminated unexpectedly.")

            if not queue.empty():
                result = queue.get()
                if type(result) == int:
                    finished[result] = True
                    continue
                logger.info(
                    f"id {result['id']} Process {result['process_id']} Loss: {result['best_loss']} String: {result['attack_string']}"
                )
                write_line_jsonl(data_save_file, result)
                results.append(result)

        for p in processes:
            p.join()

    except Exception as e:
        logger.error(f"Error: {e}")
        for p in processes:
            if p.is_alive():
                p.terminate()
        raise


if __name__ == "__main__":
    main()
