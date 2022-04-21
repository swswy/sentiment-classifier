# Artificial Neural Network
本次情感分析作业使用python语言完成，借助了keras等框架，搭建了cnn、rnn以及mlp三种神经网络
## Environment
```shell
pip install -r requirements.txt 
```
## Files
- code文件夹下为可执行代码 
    - data与preprocessdata用于预处理数据并训练得到词向量
    - hw3用于训练相关网络模型
    - evaluate用于计算模型的宏平均等指标
    - predict可输入单个句子并选择模型预测结果
- data文件夹下为实验给定的原始数据
- model文件夹下保存训练的网络模型和词向量模型
- pretreatment文件夹下保存预处理的数据
## Usage
### 训练模型
```shell
python ./code/hw3.py
```
之后可输入要训练的神经元网络类型，0为mlp，1为cnn，2为rnn，选择后若model下已存在相应的模型，则会直接加载并读取test数据测试；
否则会读取数据重新训练，而后读取test数据测试
### 评估性能
```shell
python ./code/evaluate.py
```
同上，输入要训练的神经元网络类型（注 必须相应的模型已训练），之后使用sklearn库计算相关的性能
### 单句预测
```shell
python ./code/predict.py
```
同上，输入要使用的网络类型，之后要预测的英文语句，按下回车进行预测;
输入为空时，程序结束