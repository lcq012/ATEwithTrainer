

# BaselineWithTrainer

###### 开发前的配置要求

1. 安装requirements.txt依赖库
1. 运行run.sh文件

### 文件目录说明
```
ate_baseline_trainer
├── README.md 
├── /draft/      						用于保存评估的样本（评估脚本的特性）
├── /logs/								日志文件存放地址
├── /raw_data/							数据集存放地址
├── /report/							存放最优评价指标的模型参数位置
├── bert_model.py						模型文件
├── conlleval.py						评估脚本文件
├── data_processing.py					数据处理文件
├── main.py								主函数
├── mytrainer.py						trainer函数
├── requirements.txt					依赖环境
└── run.sh								运行脚本文件

```

### 使用到的框架

- trainer

### 版本

v1.0   

初代使用trainer的baseline用于ATE任务



