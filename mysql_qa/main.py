# 导包
from cache.redis_client import RedisClient
from db.mysql_client import MySQLClient
from utils.preprocess import preprocess_text
from retrival.bm25_search import BM25Search
from base import logger
import time


# todo 1.定义MySQL问答系统类--》整合MySQL，Redis，BM25搜索功能，提供统一的查询接口
class MySQL_QA_System:
    # todo 1.1初始化
    def __init__(self):
        # 日志记录器
        self.logger = logger
        # MySQL客户端
        self.mysql_client = MySQLClient()
        # Redis客户端
        self.redis_client = RedisClient()
        # BM25搜索模型
        self.bm25_search = BM25Search(self.redis_client, self.mysql_client)

    # todo 1.2处理用户查询的核心方法--》调用BM25搜索模型，记录日志与耗时，并返回答案
    def query(self, query):
        """
        处理用户查询，通过BM25搜索从MySQL获取答案，返回结果并记录日志
        :param query:用户问题
        :return:匹配到的答案--》未找到就返回默认提示
        """
        # 记录查询开始时间
        start_time = time.time()
        # 记录info日志：标记当前正在处理的查询内容
        self.logger.info(f'正在查询：{query}')
        # 调用BM25搜索，获取答案
        # "_"表示忽略是否要进一步处理，这里我们只获取答案
        answer, _ = self.bm25_search.search(query, threshold=0.84)
        # 分支处理搜索结果 ，找到答案就记录，未找到就返回默认提示
        if answer:
            self.logger.info(f'MySQL答案：{answer}')
        else:
            self.logger.info('SQL中未找到答案，需要调用RAG')
            answer = 'SQL中未找到答案'  # 设置默认提示性答案
        # 计算查询处理耗时，并记录日志
        process_time = time.time() - start_time
        self.logger.info(f'查询耗时：{process_time:.2f}秒')
        return answer


# todo 2.主函数--》创建问答系统实例，提供交互式命令界面
def main():
    # 创建MySQL问答系统实例
    mysql_qa_system = MySQL_QA_System()
    try:
        # 打印欢迎信息和使用说明
        print("\n欢迎使用MySQL问答系统！")
        print("请输入你的问题，或者输入'exit'来结束程序。")

        # 循环获取用户输入，处理查询或者退出
        while True:
            # 获取用户输入，并移除首尾空格
            query = input("\n请输入查询：").strip()
            # 判断是否是‘退出’，如果是，就退出
            if query.lower() == 'exit':
                logger.info('退出MySQL')
                print('感谢使用MySQL问答系统！')
                break
            # 调用用户问答系统获取答案
            answer = mysql_qa_system.query(query)
            print(f'答案：{answer}')

    except Exception as e:
        print(f'系统错误：{e}')
    finally:  # 无论是否发生问题，都会走这，关闭MySQL连接
        mysql_qa_system.mysql_client.close()


# todo 3.测试代码
if __name__ == '__main__':
    main()
