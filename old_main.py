# 该脚本用于：完成EduRAG项目的整体流程，即：接受用户的问题--》BM25算阈值--》决定执行MySQL还是milvus--》获取答案
import warnings
warnings.filterwarnings("ignore")

# todo 1.导包
# 导入MySQL系统组件，用于数据库操作和搜索
from mysql_qa.cache.redis_client import RedisClient
from mysql_qa.db.mysql_client import MySQLClient
from mysql_qa.retrival.bm25_search import BM25Search
# 导入RAG系统组件，用于知识库检索和答案生成
from rag_qa.core.vector_store import VectorStore
from rag_qa.core.rag_system import RAGSystem
# 导入配置和日志工具，用于系统配置和日志记录
from base.config import Config
from base.logger import logger
# 导入OPenAI客户端，用于调用DashScope API
from openai import OpenAI
# 导入时间库，用于记录处理时间
import time


# todo 2.集成问答系统的核心类
# 该类的作用：集成问答系统的核心类，把各种工具拼接起来
# 先查MySQL中的现成答案--》靠谱就用（BM25阈值>0.85）--》不靠谱就用RAG生成--》啥都没有即没找到
class IntegratedQASystem:
    # todo 2.1初始化方法--》创建系统需要的各种工具
    def __init__(self):
        # 初始化日志和配置对象
        self.logger = logger
        self.config = Config()
        # 初始化MySQL客户端和Redis客户端，BM25搜索模块
        self.mysql_client = MySQLClient()
        self.redis_client = RedisClient()
        self.bm25_search = BM25Search(self.redis_client,self.mysql_client)
        # 初始化LLM
        try:
            # 初始化openai客户端，连接DashScope API
            self.client = OpenAI(api_key=self.config.DASHSCOPE_API_KEY, base_url=self.config.DASHSCOPE_BASE_URL)
        except Exception as e:
            # 记录openai初始化失败的错误日志
            self.logger.error(f'openai客户端初始化失败：{e}')
            # 抛出异常，停止程序运行
            raise
        # 初始化数据向量库和RAG系统
        self.vector_store = VectorStore()  # 创建向量数据库实例--》包括：向量存储和检索
        self.rag_system = RAGSystem(self.vector_store, self.call_dashscope)  # 创建RAG系统实例--》包括：检索和生成

    # todo 2.2调用大语言模型的方法：RAG
    def call_dashscope(self, prompt):
        """
        函数作用：调用Dashscope大模型生成文本--》例如：答案，子问题
        :param prompt:输入给模型的提示文本--》即：用户输入的查询问题
        :return:模型生成的文本，如果调用失败则返回错误信息
        """
        # 定义调用Dashscope API的方法，生成自然语言答案
        try:
            # 创建聊天完成请求，调用Dashscope API
            completion = self.client.chat.completions.create(
                model=self.config.MODEL_NAME,
                messages=[
                    {'role': 'system', 'content': '你是个有用的助手'},  # 系统提示
                    {'role': 'user', 'content': prompt}  # 用户输入问题
                ]
            )
            # 检查响应是否有效，返回答案内容
            return completion.choices[0].message.content if completion.choices else '错误，无效的LLM响应'
        except Exception as e:
            # 记录API调用失败的错误日志
            self.logger.error(f'LLM调用失败：{e}')
            # 返回错误信息，便于调用
            return f'错误：LLM调用失败 - {e}'

    # todo 2.3处理查询的核心方法--》用户问的问题从这进入
    def query(self, query, source_filter=None):
        """
        函数作用：处理用户查询，按步骤来，先查MySQL+BM25--》不行再RAG--》返回答案
        :param query:用户输入的查询问题
        :param source_filter:源过滤器，用于过滤来源
        :return:返回答案--》如果都没有则返回默认提示
        """
        # 记录开始时间
        start_time = time.time()
        self.logger.info(f'开始处理问题：{query}，来源过滤体条件：{source_filter}')

        # 调用BM25查MySQL中的答案，返回俩个内容：答案本身，以及是否需要RAG的表示
        answer, need_rag = self.bm25_search.search(query, threshold=0.85)

        # 根据BM25处理结果，判断
        if answer:
            # 拿到靠谱的答案
            self.logger.info(f'BM25找到靠谱答案：{answer[:50]}')
            # 算一下用了多久
            processing_time = time.time() - start_time
            self.logger.info(f'处理完成，耗时：{processing_time:.2f}秒')
            # 返回处理结果
            return answer

        elif not need_rag:
            # 缓存未命中，调用RAG重新生成
            self.logger.info(f'BM25未找到靠谱答案，调用RAG系统')
            answer = self.rag_system.generate_answer(query, source_filter=source_filter)
            self.logger.info(f'RAG生成答案：{answer[:50]}')
            # 算一下用了多久
            processing_time = time.time() - start_time
            self.logger.info(f'处理完成，耗时：{processing_time:.2f}秒')
            # 返回处理结果
            return answer
        else:
            # 没有找到答案
            self.logger.info(f'BM25未找到答案')
            # 算一下用了多久
            processing_time = time.time() - start_time
            self.logger.info(f'处理完成，耗时：{processing_time:.2f}秒')
            # 返回处理结果
            return '抱歉，我找不到答案'


# todo 3.主函数--》给用户用的交互式界面
def main():
    """
    函数功能：系统入口,用户可以在这里输入问题，选：来源过滤，然后看答案
    :return:无
    """
    qa_system = IntegratedQASystem()
    qa_system.logger.info(f'系统初始化完成，可以开始问问题了')

    try:
        # 打印欢迎信息，告诉用户怎么操作
        print("\n欢迎使用MySQL问答系统！")
        # 打印支持的学科类别
        print(f"支持的学科类别有：{qa_system.config.VALID_SOURCES}")
        # 用法
        print(f'用法：输入问题按回车键查看答案，输入exit按回车键退出系统')

        # 循环等待用户录入
        while True:
            # 接受用户录入的问题
            query = input("\n请输入问题：").strip()
            # 判断是否是exit
            if query.lower() == 'exit':
                qa_system.logger.info(f'用户输入exit，系统退出')
                print('感谢使用MySQL问答系统！')
                break
            # 让用户来选过滤来源
            source_filter = input(
                f"请输入来源过滤条件（可选,支持：{'./'.join(qa_system.config.VALID_SOURCES)}，直接回车表示不限制）").strip()
            # 判断用户录入的来源过滤对不对
            if source_filter:
                # 用户选择了过滤源-->即：过滤源非空
                if source_filter not in qa_system.config.VALID_SOURCES:
                    # 用户输入的过滤源不在支持的范围内
                    qa_system.logger.warning(f'用户输入的过滤源无效：{source_filter}')
                    # 重置为空
                    source_filter = None
                else:
                    qa_system.logger.info(f'用户选择了过滤源：{source_filter}')

            # 调用查询方法，拿到答案并提示
            answer = qa_system.query(query, source_filter=source_filter)
            print(f'答案：{answer}')

    except Exception as e:
        qa_system.logger.error(f'系统处理异常：{e}')
        print(f'处理问题时出错，请联系管理员')
    finally:
        # 不管是否出错，关闭MySQL连接，不然会占用资源
        qa_system.mysql_client.close()
        qa_system.logger.info(f'MySQ连接关闭，系统退出')


# todo 4.程序的主入口
if __name__ == '__main__':
    main()  # 启用交互式界面，处理用户输入，获取答案
