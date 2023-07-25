#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   data_processing.py
@Time    :   2023/07/22 15:00:21
@Author  :   Charlton Liu
@Version :   1.0
@Contact :   cqliunlp@gmail.com
@Desc    :   processing data
'''
import json
import torch
from tqdm import tqdm
from pdb import set_trace as stop
class ATEItem(object):
    def __init__(self, sentence, label):
        self.sentence = sentence
        self.label = label

class ATEDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, label_ids):
        self.encodings = encodings
        self.label_ids = label_ids

    def __getitem__(self, id):
        item = {key: torch.tensor(val[id]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.label_ids[id])
        return item
    
    def __len__(self):
        return len(self.label_ids)

class ATEProcessor():
    def load_dataset(self, file_path: str):
        with open(file_path, 'r') as f:
            data = json.load(f)
        items = []
        for item in data:
            sent, label = item[0], item[1]
            assert len(sent.split()) == len(label.split())
            items.append(ATEItem(sentence=sent, label=label))
        return items
    
    def label2id(self, labels):
        result = []
        for label in labels.split():
            if label.startswith("O"):
                result.append(0)
            elif label.startswith("B"):
                result.append(1)
            elif label.startswith("I"):
                result.append(2)
        return result

    def convert_examples_to_features(self, args, tokenizer, examples):
        encodings = {}
        encodings["input_ids"] = []
        encodings["attention_mask"] = []
        labels = []
        for example in tqdm(examples, total = len(examples), desc="Processing ATE Dataset"):
            tokenizer_inputs = tokenizer(
            example.sentence.split(),
            max_length=args.max_length,
            add_special_tokens=True,
            padding = 'max_length',
            truncation=True,
            is_split_into_words = True
            )
            encodings["input_ids"].append(tokenizer_inputs["input_ids"])
            encodings["attention_mask"].append(tokenizer_inputs["attention_mask"])
            raw_label = self.label2id(example.label)
            # 对文本中的标签进行处理来对其分词之后的单词
            word_ids = tokenizer_inputs.word_ids() 
            previous_word_idx = None
            label_ids = []
            try:
                for word_idx in word_ids:
                    # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                    # ignored in the loss function.
                    if word_idx is None:
                        label_ids.append(-100)
                    # We set the label for the first token of each word.
                    elif word_idx != previous_word_idx:
                        label_ids.append(raw_label[word_idx])
                    # For the other tokens in a word, we set the label to either the current label or -100, depending on
                    # the label_all_tokens flag.
                    else:
                        label_ids.append(raw_label[word_idx])
                    previous_word_idx = word_idx
            except IndexError:
                stop()
            labels.append(label_ids)
        return ATEDataset(encodings, labels)


def load_examples(args, tokenizer, type='train'):
    ATEProcessing = ATEProcessor()
    if type == "train":
        examples = ATEProcessing.load_dataset(args.train_file)
    elif type =="dev":
        examples = ATEProcessing.load_dataset(args.dev_file)
    elif type == "test":
        examples = ATEProcessing.load_dataset(args.test_file)
    return ATEProcessing.convert_examples_to_features(args, tokenizer, examples)