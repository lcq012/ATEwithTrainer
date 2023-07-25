#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   bert.py
@Time    :   2023/07/23 00:15:00
@Author  :   Charlton Liu
@Version :   1.0
@Contact :   cqliunlp@gmail.com
@Desc    :   None
'''
from transformers import trainer
from pdb import set_trace as stop
import logging
from typing import Union
import logging
logger = trainer.logger

class MyTrainer(trainer.Trainer):
    def __init__(self, model, args, train_dataset, eval_dataset, tokenizer, compute_metrics, log_file: Union[str, None]):
        # 配置日志记录器
        super().__init__(
            model=model, 
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics) 
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(log_file)
        # 创建格式化器并设置格式
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        # 将文件处理器添加到logger对象
        logger.addHandler(file_handler)


    def compute_loss(self, model, inputs, return_outputs=False):
        results = model(**inputs)
        loss = results['loss']
        inputs = results['input_ids']
        return (loss, results) if return_outputs else loss

    
    def get_logger(self):
        return logger