import torch
from api_secrets import get_api_key_by_name
from argparse import ArgumentParser
from constants import SAVE_EVERY
from dataset_utils import get_dataset_info, get_questions_with_exemplars
from experiment_config import ExperimentConfig
from experiment_saver import ExperimentSaver
from itertools import permutations
from models import get_model_by_name
from tqdm import tqdm
from parallelbar import progress_map
import more_itertools as mit
from transformers import AutoTokenizer

# import multiprocessing
# import torch.multiprocessing as mp
# multiprocessing.set_start_method("spawn")

import copy


def get_config_and_api_key_name_from_args():
    parser = ArgumentParser()
    parser.add_argument("ds_name", help="Dataset name")
    parser.add_argument("model_name", help="Model name")
    parser.add_argument("style_name", help="Style name")
    parser.add_argument("n_shots", type=int, help="# of shots")
    parser.add_argument(
        "--do_strong_shuffle",
        action="store_true",
        help="Force correct answer index to change for each example",
    )
    parser.add_argument(
        "--do_perm",
        action="store_true",
        help="Process every example with all possible answer orderings",
    )
    args = parser.parse_args()
    args = vars(args)
    return ExperimentConfig(**args)


def the_work(inputs):
    rank, model1, style_name, input_tuples = inputs
    device = torch.device(f"cuda:{rank}")
    model1.model.to(device)
    # Load model
    model_call = {
        "natural": model1.process_question_natural,
        "brown": model1.process_question_brown,
    }[style_name]
    results = []
    for idx, query in input_tuples:
        response = model_call(query)
        results.append((idx, query, response))
    return results


def run_experiment(config):

    # Get API key
    # api_key = get_api_key_by_name(name=api_key_name)

    # Get questions with exemplars
    qwes = get_questions_with_exemplars(
        info=get_dataset_info(config.ds_name),
        n_shots=config.n_shots,
        do_strong_shuffle=config.do_strong_shuffle,
    )

    # Run experiment, saving results
    saver = ExperimentSaver(save_fname=config.get_save_fname())
    # if config.do_perm:
    #         for perm_order in permutations(range(qwe.get_n_choices())):
    #             qwe_copy = copy.deepcopy(qwe)
    #             qwe_copy.permute_choices(perm_order)
    #             response = model_call(qwe_copy)
    #             saver["question_idx"].append(q_idx)
    #             saver["perm_order"].append(perm_order)
    #             saver["qwe"].append(vars(qwe_copy))
    #             saver["model_response"].append(vars(response))

    #         # When doing permutations we ignore SAVE_EVERY and
    #         # save after every question
    #         saver.save()

    work_items = [(q_idx, qwe) for q_idx, qwe in enumerate(qwes)]
    NUM_PROCESS = 6
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    standard = get_model_by_name(config.model_name, tokenizer)
    divid = [
        (rank, copy.deepcopy(standard), config.style_name, x)
        for rank, x in enumerate(
            mit.chunked(work_items, (int)(len(work_items) / NUM_PROCESS))
        )
    ]
    results, return_failed_tasks = progress_map(
        the_work, divid, context="spawn", return_failed_tasks=True
    )
    print(return_failed_tasks)
    results = sum(results, [])
    results = sorted(results, key=lambda x: x[0])  # 3 is the real input, 0 is the idx

    for q_idx, qwe, response in tqdm(results):
        saver["question_idx"].append(q_idx)
        if qwe.task is not None:
            saver["task"].append(qwe.task)
        saver["qwe"].append(vars(qwe))
        saver["model_response"].append(vars(response))

    saver.save()


if __name__ == "__main__":
    run_experiment(get_config_and_api_key_name_from_args())
