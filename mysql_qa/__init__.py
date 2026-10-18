import os,sys

# 记录当前目录
current_dir = os.path.abspath(__file__)
# print(current_dir)

# 记录mysql_qa_path目录--》MySQL版的问答系统目录
mysql_qa_path = os.path.dirname(current_dir)
# print(mysql_qa_path)

# 添加mysql_qa_path到系统路径中
sys.path.insert(0, mysql_qa_path)

# 具体导包动作
from db.mysql_client import MySQLClient # MySQL客户端
from cache.redis_client import RedisClient  # Redis客户端
from retrival.bm25_search import BM25Search # BM25 --> 相似度检索


