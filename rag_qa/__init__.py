import os,sys

# 记录当前目录
current_dir = os.path.abspath(__file__)
# print(current_dir)

# 记录rag_qa_path目录--》MySQL版的问答系统目录
rag_qa_path = os.path.dirname(current_dir)
# print(rag_qa_path)

# 添加rag_qa_path到系统路径中
sys.path.insert(0, rag_qa_path)

# 具体导包动作
from core.prompts import RAGPrompts         # RAG系统提示词
from core.vector_store import VectorStore   # 向量存储
# from core.rag_system import RAGSystem       # RAG核心代码--》没有添加历史记录和流式输出时选择
from core.new_rag_system import RAGSystem     # 添加历史记录和流式输出时选择


