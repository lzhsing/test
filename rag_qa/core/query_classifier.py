# 该脚本用于: 基于BERT模型实现查询分类器, 将用户查询分为'通用知识'和'专业咨询'两大类.
# 核心流程: 加载预训练BERT模型 -> 数据预处理 -> 模型训练 -> 模型评估 -> 预测查询类别
# 应用场景: 在RAG系统中, 可根据分类结果决定是否需要检索专业文档(例如: 专业咨询可检索文档, 通用知识由模型直接回复即可)


# todo 1.导包
# 导入标准库
import json
import os, sys

# 路径配置: 解决跨目录模块导入问题 -> 定位项目各层级目录.
current_dir = os.path.dirname(os.path.abspath(__file__))
rag_qa_path = os.path.dirname(current_dir)
project_root = os.path.dirname(rag_qa_path)
bert_path = os.path.join(rag_qa_path, 'models', 'bert-base-chinese')
# 将项目目录添加到系统路径中
sys.path.insert(0, project_root)

# 导入 PyTorch
import torch
# 导入日志
from base.logger import logger
# 导入numpy
import numpy as np
# 导入 Transformers 库
from transformers import BertTokenizer, BertForSequenceClassification  # BERT分词器和序列分类模型.
# Trainer: 负责实际执行训练过程(用模型, 数据和 TrainingArguments的配置跑训练) -> 相当于 训练任务的执行者, 按照 TrainingArguments(说明书)具体干活
# TrainingArguments: 负责定义训练的各种参数和配置(例如: 训练多久, 多大批次, 保存到哪等...) -> 相当于 训练任务的说明书,提前写好训练的各种规则和细节.
from transformers import Trainer, TrainingArguments
# 导入train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# todo 2.定义QueryClassier类：封装BERT查询分类的完整流程--》模型加载，训练，评估，预测...
class QueryClassifier:
    # todo 2.1 定义初始化方法，配置模型路径，加载分词器，选择设备，定义标签映射
    def __init__(self, model_path='../models/bert_query_classifier'):
        # 1. 存储模型路径: 用于后续加载或保存模型.
        self.model_path = model_path
        # 2.加载BERT分词器：将文本转成模型可以理解的输入
        # 2.1拼接预训练BERT模型的本地路径
        bert_path = os.path.join(rag_qa_path, 'models', 'bert-base-chinese')
        # 2.2加载分词器
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        # 3.初始化模型变量，后续通过load_model()加载或创建模型
        self.model = None
        # 4.选择设备：GPU或CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 5.记录日志
        logger.info(f'使用设备: {self.device}')
        # 6.定义标签映射：将标签映射为模型输出的索引，级：通用知识 -> 0, 专业咨询 -> 1
        self.label_map = {"通用知识": 0, "专业咨询": 1}
        # 加载模型，初始化时自动调用
        self.load_model()

    # todo 2.2加载模型--》从指定路径加载模型，若无则创建新模型
    def load_model(self):
        """
        函数功能：加载模型，优先从self.model_path加载，不存在则初始化基于Bert-base-Chinese的新模型
        :return:
        """
        # 检查模型路径是否存在--》即：是否有已经训练的模型
        if os.path.exists(self.model_path):
            # 加载预训练模型
            self.model = BertForSequenceClassification.from_pretrained(self.model_path)
            # 将模型移动到指定设备
            self.model.to(self.device)
            logger.info(f'模型已加载: {self.model_path}')
        else:
            # 若模型不存在，初始化新模型
            # 参1：模型路径  参2：二分类任务
            self.model = BertForSequenceClassification.from_pretrained(bert_path, num_labels=2)
            # 将模型移动到指定设备
            self.model.to(self.device)
            logger.info(f'初始化新模型BERT新模型')

    # todo 2.3保存模型的方法--》将训练好的模型和分词器保存储到本地
    def save_model(self):
        self.model.save_pretrained(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)
        logger.info(f'模型已保存至{self.model_path}')

    # todo 2.4数据预处理方法--》将文本标签转化成BERT模型所需的输入格式
    def preprocess_data(self, texts, labels):
        """
        函数功能：对文本进行分词，截断，填充，将标签转换为：数字
        :param texts: 待处理的文本列表
        :param labels: 文本对应的标签列表
        :return: 处理后的编码（input_ids,attention_mask）和数字标签列表
        """
        # 使用BERT分词器将文本转换成模型输入格式
        embeddings = self.tokenizer(
            texts,
            truncation=True,  # 超过最大长度时截断
            padding=True,  # 填充
            max_length=128,  # 最大长度
            return_tensors='pt'  # 返回张量
        )
        # 将文本标签转换成数字
        return embeddings, [self.label_map[label] for label in labels]

    # todo 2.5创建数据集方法：自定义DataSet类，封装：编码和标签(Tensor--》TensorDataSet--》TensorDataloader--》训练模型)
    def create_dataset(self, embeddings, labels):
        # 定义内部DataSet类，继承自torch.utils.data.Dataset
        class DataSet(torch.utils.data.Dataset):
            # 初始化方法
            def __init__(self, encodings, labels):
                super().__init__()
                self.encodings = encodings  # 文本编码，格式：（input_ids, attention_mask）
                self.labels = labels  # 数字标签列表

            # 根据索引获取单条数据
            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item

            # 返回数据集长度
            def __len__(self):
                return len(self.labels)

        # 返回实例化的DataSet对象
        return DataSet(embeddings, labels)

    # todo 2.6训练模型方法--》加载数据集，预处理，配置训练参数并训练模型
    def train_model(self, data_file='..\classify_data\完整数据.txt'):
        """
        函数功能：训练BERT分类模型，区分查询分类为’通用知识‘和’专业咨询‘
        :param data_file:数据集文件路径
        :return:无
        """
        # 检查数据集文件是否存在
        if not os.path.exists(data_file):
            logger.error(f'数据集文件{data_file}不存在 ')
            raise FileNotFoundError(f'数据集文件{data_file}不存在')

        # 加载数据集，从json文件中读取查询文本和对应的标签
        with open(data_file, 'r', encoding='utf-8') as f:
            data = [json.loads(value) for value in f.readlines()]

        # 提取文本和标签，分离查询文本和对应的分类标签
        texts = [item['query'] for item in data]
        labels = [item['label'] for item in data]

        # 划分训练集和测试集
        train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2,
                                                                              random_state=42)

        # 数据预处理：将文本和标签转换成BERT模型所需的输入格式--》调用自定义的preprocess_data方法
        train_encodings, train_labels = self.preprocess_data(train_texts, train_labels)
        test_encodings, test_labels = self.preprocess_data(test_texts, test_labels)

        # 创建数据集对象，将编码和标签封装成Pytorch可识别的DataSet对象
        train_dataset = self.create_dataset(train_encodings, train_labels)
        test_dataset = self.create_dataset(test_encodings, test_labels)

        # 设置训练参数
        training_args = TrainingArguments(
            output_dir='./bert_results',  # 模型保存目录--》 './bert_results'
            num_train_epochs=3,  # 训练轮数
            per_device_train_batch_size=8,  # 每设备训练批次大小
            per_device_eval_batch_size=8,  # 每设备评估批次大小
            warmup_steps=20,  # 预热步数,训练初期学习率从0逐步增加到设定值
            weight_decay=0.01,  # 权重衰减，防止过拟合
            logging_dir='./bert_logs',  # 日志保存目录
            logging_steps=10,  # 日志记录步数,即：每10步记录一次日志
            evaluation_strategy='epoch',  # 评估策略：每轮结束评估
            save_strategy='epoch',  # 保存策略：每轮结束保存
            load_best_model_at_end=True,  # 是否加载最佳模型
            save_total_limit=1,  # 保存模型总数限制，即：最多保存1个模型,超出时自动删除旧的
            metric_for_best_model='eval_loss',  # 最佳模型评估指标
            fp16=False  # 禁用fp16，使用fp32
        )

        # 初始化trainer：封装模型，训练参数，数据集，评估指标计算方法
        trainer = Trainer(
            model=self.model,  # 待训练的模型
            args=training_args,  # 训练参数
            train_dataset=train_dataset,  # 训练数据集
            eval_dataset=test_dataset,  # 测试数据集
            compute_metrics=self.compute_metrics  # 自定义的评估指标（准确率）
        )

        # 训练模型
        logger.info('开始训练模型...')
        trainer.train()
        self.save_model()

        # 评估模型
        self.evaluate_model(test_texts, test_labels)

    # todo 2.7计算评估指标方法--》定义计算模型在验证集上的准确率
    def compute_metrics(self, eval_pred):
        """
        计算分类任务的评估指标--》准确率
        :param eval_pred:包含模型输出logits和真实标签的元组
        :return:准确率字典
        """
        # 模型输出logits和真实标签
        logits, labels = eval_pred
        # 取logits最大值对应的索引作为预测结果
        predictions = np.argmax(logits, axis=-1)
        # 计算准确率
        accuracy = (predictions == labels).mean()
        # 返回准确率字典
        return {'accuracy': accuracy}

    # todo 2.8评估模型方法--》生成分类报告和混淆矩阵，全面分析模型性能
    def evaluate_model(self, texts, labels):
        """
        在给定的文本和标签上评估模型，并生成分类报告和混淆矩阵
        :param texts: 待评估的文本列表
        :param labels: 真实标签列表（数字形式）
        """
        # 对文本进行编码，仅处理文本，因为标签已经转换成数字形式
        encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors='pt')

        # 创建评估数据集
        eval_dataset = self.create_dataset(encodings, labels)

        # 使用trainer进行预测
        trainer = Trainer(model=self.model)
        predictions = trainer.predict(eval_dataset)

        # 解析预测结果:取logits最大值对应的索引作为预测结果
        pred_labels = np.argmax(predictions.predictions, axis=-1)
        true_labels = labels

        # 记录评估结果到日志
        logger.info('分类报告：')
        logger.info(classification_report(
            true_labels,
            pred_labels,
            target_names=self.label_map.keys()
        ))
        # 混淆矩阵：展示预测标签和真实标签的匹配情况
        logger.info('混淆矩阵：')
        logger.info(confusion_matrix(true_labels, pred_labels))

    # todo 2.9预测类别方法--》对单个查询文本进行分类预测
    def predict_category(self, text):
        """
        对单个查询文本进行分类预测--》通用知识或专业咨询
        :param text: 待分类的查询文本
        :return: 预测类别，如'通用知识'或'专业咨询'
        """
        # 检查模型是否加载
        if self.model is None:
            # 模型未加载
            logger.error('模型未加载，请先加载模型')
            return '通用知识'

        # 对查询文本进行编码
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors='pt'
        )
        # 将编码数据迁移到指定设备
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        # 模型推理--》不计算梯度，提高效率
        with torch.no_grad():
            # 获取模型输出--》包含logits
            outputs = self.model(**encoding)
            # 获取预测结果--》logits最大值对应的索引
            prediction = torch.argmax(outputs.logits, dim=1).item()

        # 返回预测结果标签
        return '专业咨询' if prediction == 1 else '通用知识'


# todo 3.测试代码
if __name__ == '__main__':
    # 实例化查询分类器--》自动加载模型，若无则初始化新模型
    query_classifier = QueryClassifier()

    # 训练模型
    # data_file = r'..\classify_data\完整数据.txt'
    # query_classifier.train_model(data_file)

    # 实例预测
    text = '如何使用BERT进行文本分类？'
    category = query_classifier.predict_category(text)
    print(f'文本: {text} -> 类别: {category}')
