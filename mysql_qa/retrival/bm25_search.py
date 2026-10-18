# BM25搜索，阈值》=0.85才会走MySQL

# 导包
from rank_bm25 import BM25Okapi
import numpy as np
# 导入文本预处理
from mysql_qa.utils.preprocess import preprocess_text
# 导入日志
from base import logger
from mysql_qa.db.mysql_client import MySQLClient
from mysql_qa.cache.redis_client import RedisClient


# todo 1.定义BM25搜索类--》封装数据加载，模型初始化，相似度计算，答案检索
class BM25Search:
    # todo 1.1初始化函数--》客户端实例，模型参数，加载数据
    def __init__(self, redis_client, mysql_client):
        # 日志实例化
        self.logger = logger
        # Redis客户端实例
        self.redis_client = redis_client
        # MySQL客户端实例
        self.mysql_client = mysql_client
        # 初始化BM25模型
        self.bm25 = None
        # 初始化分此后的问题列表
        self.questions = None
        # 初始化分词后的问题列表
        self.tokenized_questions = None
        # 初始化原始问题列表
        self.original_questions = None
        # 调用数据加载方法--》加载问题数据，并初始化模型
        self._load_data()  # 加"_"的函数表示内置函数

    # todo 1.2数据加载方法--》优先从Redis缓存加载，缓存未命中则从MySQL中加载
    def _load_data(self):
        # 定义redis缓存键--》区分原始问题和分词后问题的缓存
        original_key = 'qa_original_questions'
        tokenized_key = 'qa_tokenized_questions'

        # 从Redis缓存中获取原始问题--》缓存读取速度快，减少数据库访问
        self.original_questions = self.redis_client.get_data(original_key)
        # print(self.original_questions)

        # 从Redis中获取分词后问题
        self.tokenized_questions = self.redis_client.get_data(tokenized_key)
        # print(self.tokenized_questions)

        # 若Redis缓存中无数据（或数据不完整）--》则从MySQL中加载并更新数据
        if not self.original_questions or not self.tokenized_questions:
            # 从MySQL中获取原始问题数据列表，格式为元组嵌套，如[(q1,), (q2,), ...]
            self.original_questions = self.mysql_client.fetch_questions()
            # print(self.original_questions)
            # 若MySQL也无问题数据，记录警告日志
            if not self.original_questions:
                self.logger.warning("MySQL中无问题数据")
                return

            # 对原始问题进行分词处理
            tokenized_questions = [preprocess_text(i[0]) for i in self.original_questions]

            # 将原始问题列表转成字符串--》去除元组嵌套，并缓存到redis中
            self.redis_client.set_data(original_key, [(q[0]) for q in self.original_questions])
            self.redis_client.set_data(tokenized_key, tokenized_questions)

        # 保存分词后的列表--》用于BM25模型初始化
        self.questions = self.tokenized_questions
        # 初始化BM25模型
        self.bm25 = BM25Okapi(self.questions)
        # 记录info日志
        self.logger.info("BM25模型初始化完成")

    # todo 1.3softmax分数归一化方法--》将BM25分数转换成概率分布，便于阈值判断
    def _softmax(self, scores):
        # 对输入分数进行softmax归一化，输出总和为1的概率分布
        # 计算指数分数：每个分数减去最大值--》防止数值过大导致溢出
        exp_scores = np.exp(scores - np.max(scores))
        # 返回归一化结果
        return exp_scores / np.sum(exp_scores)

    # todo 1.4核心搜索方法--》处理查询，计算相似度，返回匹配答案或提示无结果
    def search(self, query, threshold=0.85):
        """
        根据输入查询检索最相似的问题，并返回对应的答案
        :param query:用户查询问题
        :param threshold:相似度阈值--》超过此值即为匹配成功
        :return:匹配成功：（答案，True）；匹配失败：（None，False）        True：缓存命中
        """
        # 检查查询有效性--》空或者非字符串类型认为无效
        if not query or not isinstance(query, str):
            self.logger.warning("无效的查询")
            return None, False
        # 到这非空，先检查Redis缓存--》缓存命中则直接返回
        cache_answer = self.redis_client.get_answer(query)
        if cache_answer:
            # 缓存命中，返回结果
            return cache_answer, True
        # 未命中，需要从数据库中查询
        try:
            # 对查询文本进行预处理--》分词，和问题库预处理格式一致
            query_tokens = preprocess_text(query)
            # 计算查询问题与（数据库中）所有问题的BM25相似度
            scores = self.bm25.get_scores(query_tokens)
            # print(scores)
            # 对分数进行归一化处理--》转换成概率分布
            softmax_score = self._softmax(scores)
            # print(softmax_score)
            # 找到最高相似度对应的索引和分数
            best_idx = np.argmax(softmax_score)
            best_score = softmax_score[best_idx]
            # print(f"Best Question: {self.original_questions[best_idx]}")

            # 若最高分超过阈值视为匹配成功
            if best_score >= threshold:
                # 获取最高匹配度的原始问题--》用于从数据库查询答案
                original_question = self.original_questions[best_idx]
                # 从MySQL数据库中查询该问题的答案
                answer = self.mysql_client.fetch_answer(original_question)
                # 返回答案
                if answer:
                    # 将查询答案对缓存到Redis中-->供下次查询用
                    self.redis_client.set_data(f'answer:{query}', answer)
                    # 记录info日志
                    self.logger.info(f"查询成功：{best_score:.3f}")
                    return answer, True
            # 若未超过阈值记录info日志并返回结果
            self.logger.info(f"未找到答案，最高匹配度：{best_score:.3f}")
            return None, False  # False：未命中，需要走RAG

        except Exception as e:
            self.logger.error(f"数据库查询异常：{e}")
            return None, False  # False代表Redis中没查到


# todo 2.测试代码
if __name__ == '__main__':
    # 实例化Redis和MySQL客户端
    redis_client = RedisClient()
    mysql_client = MySQLClient()

    # 创建BM25搜索实例
    bm25_search = BM25Search(redis_client, mysql_client)

    # 测试搜索功能
    result = bm25_search.search(query='win10如何安装python虚拟环境以及爬虫环境')
    print(result)
