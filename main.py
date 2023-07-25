#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2023/07/22 15:07:27
@Author  :   Charlton Liu
@Version :   1.0
@Contact :   cqliunlp@gmail.com
@Desc    :   None
'''
import argparse
import torch
import random
import numpy as np
from data_processing import load_examples
from transformers import (
    AutoTokenizer,
    TrainingArguments,
)
import codecs
from pdb import set_trace as stop
from bert_model import TokenClassification
from mytrainer import MyTrainer
import os
tokenizer = None
convall_file = None
args = None
def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.
    Args:
        seed (`int`):
            The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def my_metric_func(eval_preds):
    id2label = {0:'O', 1:'B-AS', 2:'I-AS'}
    from pdb import set_trace as stop
    with codecs.open(convall_file, "w", encoding="utf-8") as writer:
        logits, input_ids = eval_preds.predictions[0], eval_preds.predictions[2]
        labels = eval_preds.label_ids
        predicts = np.argmax(logits, axis=-1).tolist()
        special_tokens = tokenizer.all_special_tokens
        for input_id, label, predict in zip(input_ids, labels, predicts):
            tokens = tokenizer.convert_ids_to_tokens(input_id)
            for idx in range(len(tokens)):
                if tokens[idx] not in special_tokens:
                    cur_label = id2label[label[idx]]
                    pre_label = id2label[predict[idx]]
                    writer.write(tokens[idx]+' '+cur_label+' '+pre_label+'\n')
            writer.write('\n')
    from conlleval import return_report
    eval_result, p, r, f = return_report(convall_file)
    try:
        file_name = args.dataset_name + "-" + str(args.seed) + ".txt"
        with open(os.path.join(args.convall_file, file_name), "a+", encoding="utf8") as report:
            report.write(''.join(eval_result))
            report.write("#" * 80 + "\n")
    except:
        raise
    # 返回一个字典，其中键是评价指标的名称，值是评价指标的值
    return {"precision": p, "recall":r, "f1":f}


def parse_args():
    """
    Function: To load parameters
    
    """
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on ATE"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="The seeds for model"
    )
    parser.add_argument(
        "--convall_file",
        type=str,
        default=None,
        help="The name of convall_file",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--dev_file", type=str, default=None, help="A csv or a json file containing the dev data."
    )
    parser.add_argument(
        "--test_file", type=str, default=None, help="A csv or a json file containing the test data."
    )
    parser.add_argument(
        "--logging_dir", type=str, default=None, help="A directory for logging."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--plm_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--dev_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the test dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--eval_steps", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument("--num_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument("--num_labels", type=int, default=3, help="The number of labels.")
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="ate",
        help="The name of the task.",
    )
    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None:  
        raise ValueError("Need a task name.")
    elif args.train_file is None:
        raise ValueError("Need a train name.")
    elif args.dev_file is None:
        raise ValueError("Need a dev name.")
    elif args.test_file is None:
        raise ValueError("Need a test name.")

    return args


def main():
    global tokenizer
    global convall_file
    global args
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.plm_path)
    convall_file = os.path.join(args.convall_file, args.dataset_name+'_'+str(args.seed)+'.txt')
    # 准备训练参数
    training_args = TrainingArguments(
        output_dir='./report',
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.dev_batch_size,
        learning_rate=args.learning_rate,
        eval_steps=args.eval_steps,
        warmup_steps=100,
        logging_dir=args.logging_dir,
        evaluation_strategy='steps',
        load_best_model_at_end=True,
        seed=args.seed,
        metric_for_best_model='f1',
        greater_is_better=True,
        weight_decay=0.01
    )
    # 训练集加载成dataloader
    train_data = load_examples(args, tokenizer, 'train')
    dev_data = load_examples(args, tokenizer, 'dev')
    test_data = load_examples(args, tokenizer, 'test')
    model = TokenClassification.from_pretrained(args.plm_path, num_labels=args.num_labels)
    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=dev_data,
        tokenizer=tokenizer,
        compute_metrics=my_metric_func,
        log_file=os.path.join('./logs/', args.dataset_name+'_'+str(args.seed)+'.log')
    )
    trainer.train()
    # 对测试集进行测试
    test_predictions = trainer.predict(test_data)
    logger = trainer.get_logger()
    logger.info('---------------------------------')
    logger.info(f'When the random seed is ${args.seed}, the best evaluation metrics in the ${args.dataset_name} dataset is {test_predictions.metrics}')
    logger.info('---------------------------------')

if __name__ == "__main__":
    main()