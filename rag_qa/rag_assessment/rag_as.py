# RAGAS框架对 RAG系统做评分，主要考察：检索和生成部分
# 4个指标
# 检索部分：上下文精确率，上下文召回率
# 生成部分：答案忠实度，答案相关性

# 导包
import pandas as pd
# 导入ragas库的evaluate函数，执行RAG评估
from ragas import evaluate
# 导入ragas库的metrics模块，包含忠实度、相关性等指标
from ragas.metrics import (
    context_precision,      # 上下文精确率
    context_recall,         # 上下文召回率
    faithfulness,           # 答案忠实度
    answer_relevancy        # 答案相关性
)
# 导入datasets库的Dataset类，用于构建RAGAS所需的数据格式
from datasets import Dataset
# 导入json库，用于加载json格式的数据评估集
import json
import os

# 导入langchain_openai的嵌入模型和聊天模型，用于评估时的语义计算和推理
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# 这里用ollama
from langchain_ollama import ChatOllama,OllamaEmbeddings

# todo 1.加载评估数据集--》读取预先生成的包含问题，答案，上下文，真实答案的json文件
with open('rag_evaluate_data.json','r',encoding='utf-8') as f:
    data = json.load(f)
print(len(data))

# todo 2.转换数据格式--》将原始数据转成RAGAS所需的数据格式（DataSet）
# 构建符合RAGAS数据格式的字典，key：固定的字段名，value：对应的数据列表
eval_data = {
    'question':[item['question'] for item in data],             # 问题列表，格式为：['问题1','问题2',...]
    'answer':[item['answer'] for item in data],             # 答案列表，格式为：['答案1','答案2',...]
    'contexts':[item['context'] for item in data],             # 上下文列表，格式为：['上下文1','上下文2',...]
    'ground_truth':[item['ground_truth'] for item in data]     # 真实答案列表，格式为：['真实答案1','真实答案2',...]
}
# 将字典转成DataSet对象
dataset = Dataset.from_dict(eval_data)
print(f'转换后的数据集：{dataset}')

# todo 3.配置评估模型--》选择用于评估的LLM和嵌入模型（用于计算相似度）
# 1.使用OpenAI模型
llm = ChatOpenAI(
    model='deepseek-v4.1-flash',
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=os.environ['DASHSCOPE_API_KEY']
)
# 初始化OpenAI嵌入模型，用于计算语义相似度
embeddings = OpenAIEmbeddings(
    model='deepseek-v4.1-flash',
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=os.environ['DASHSCOPE_API_KEY'],
    check_embedding_ctx_length=False,
    chunk_size=10
)

# # 2.使用Ollama模型
# # 初始化Ollama聊天模型
# llm = ChatOllama(model='qwen3.5:9b',base_url='http://localhost:11434')
# # 初始化Ollama嵌入模型，用于计算语义相似度
# embeddings = OllamaEmbeddings(model='qwen3.5:9b',base_url='http://localhost:11434')

answer_relevancy.strictness = 1.0
# todo 5.执行评估--》使用RAGAS库的evaluate函数，传入数据集，LLM和嵌入模型，返回评估结果
results = evaluate(
    dataset=dataset,
    llm=llm,
    embeddings=embeddings,
    metrics=[
        context_precision,      # 上下文精确率
        context_recall,         # 上下文召回率
        faithfulness,           # 答案忠实度
        answer_relevancy
    ]
)
print(results)
print(results.to_pandas())






