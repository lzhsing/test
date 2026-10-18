# 基于old_main的优化，在其基础上添加了：历史对话+流式输出
import warnings

warnings.filterwarnings("ignore")

# todo 1.导包
# 导入MySQL系统组件，用于数据库操作和搜索
from mysql_qa import RedisClient, MySQLClient, BM25Search
# 导入RAG系统组件，用于知识库检索和答案生成
from rag_qa import VectorStore, RAGSystem
# 导入配置和日志工具
from base.logger import logger
from base.config import Config
# 导入OpenAI客户端，用于调用Dashscope API
from openai import OpenAI
# 导入时间库
import time
import pymysql  # python操作MySQL的库，用于数据库操作的异常捕获，增删改查等操作
import uuid  # 生成唯一会话ID，用于标识：用户会话


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
        self.bm25_search = BM25Search(self.redis_client, self.mysql_client)
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

        # 优化1
        # 初始化对话历史表：在MySQL中创建存储会话记录的表
        self.init_conversation_table()

    # 优化2：初始化对话历史表的方法：在MySQL中创建存储对话记录的表结构
    def init_conversation_table(self):
        """
        初始化MySQL中的conversations表，用于存储对话历史
        :return:
        """
        try:
            # 执行SQL语句--》创建conversation表，包含会话ID，问题，答案和时间戳
            self.mysql_client.cursor.execute("""
                create table if not exists conversations(
                    id int auto_increment primary key,
                    session_id varchar(36) not null,
                    question text not null,
                    answer text not null,
                    timestamp datetime not null,
                    index idx_session_id(session_id)
                )engine=innodb 
            """)
            # 提交数据库事务
            self.mysql_client.connection.commit()
            # 记录初始化表成功日志
            self.logger.info(f'对话历史表初始化成功')
        except pymysql.MySQLError as e:
            # 记录初始化表失败的错误日志
            self.logger.error(f'对话历史表初始化失败：{e}')
            raise

    # todo 2.2调用大语言模型的方法：RAG
    def call_dashscope(self, prompt):
        """
        函数作用：调用Dashscope大模型生成文本--》流式输出
        :param prompt:输入给模型的提示文本，包含用户问题及上下文
        :return:生成器，逐段返回模型生成的文本
        """
        # 定义调用Dashscope API的方法，生成自然语言答案
        try:
            # 创建聊天完成请求，调用Dashscope API，启用流式输出，设置超时时间
            completion = self.client.chat.completions.create(
                model=self.config.MODEL_NAME,
                messages=[
                    {'role': 'system', 'content': '你是个有用的助手'},  # 系统提示
                    {'role': 'user', 'content': prompt}  # 用户输入问题
                ],
                stream=True,
                timeout=60  # 设置超时时间为60秒
            )
            # 遍历流式输出的每个chunk
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    # 获取当前chunk的内容
                    content = chunk.choices[0].delta.content
                    # 逐token返回，实现流式输出
                    yield content
        except Exception as e:
            # 记录API调用失败的错误日志
            self.logger.error(f'API调用失败：{e}')
            # 返回错误信息
            return f'错误：API调用失败 - {e}'

    # todo 2.4获取最近对话历史的内部方法：从MySQL中查询指定会话（id）的最近5轮对话
    def _fetch_recent_history(self, session_id):
        """
        函数作用：从MySQL中查询指定会话（id）的最近5轮对话
        :param session_id:会话ID，用于标识用户
        :return:包含对话历史的列表，每轮对话是一个字典，包含问题和答案
        """
        try:
            # 执行SQL查询，从conversation表中获取指定会话的最近5轮对话
            self.mysql_client.cursor.execute("""
                select question, answer from conversations where session_id = %s order by timestamp desc limit %s
            """, (session_id, 5))
            # 将查询结果转换为字典列表
            history = [{'question': row[0], 'answer': row[1]} for row in self.mysql_client.cursor.fetchall()]
            # 反转列表，使时间顺序正确
            return history[::-1]
        except pymysql.MySQLError as e:
            # 记录查询失败的错误日志
            self.logger.error(f'对话历史查询失败：{e}')
            return []

    # todo 2.5获取会话历史的方法：对外提供获取指定会话历史的接口
    def get_conversation_history(self, session_id):
        """
        函数作用：获取指定会话历史的对话历史
        :param session_id:会话ID，用于标识用户
        :return:列表，包含最近5轮会话历史
        """
        return self._fetch_recent_history(session_id)

    # todo 2.6更新会话历史的方法：将新对话记录存入MySQL，并保留最近5轮
    def update_session_history(self, session_id: str, question: str, answer: str) -> list:
        """
        函数作用：将新对话记录存入MySQL，并保留最近5轮
        :param session_id:会话唯一标识
        :param question:用户问题
        :param answer:系统生成的答案
        :return:
        """
        try:
            # 插入新的会话记录，将问题，答案，及当前时间存入数据库
            self.mysql_client.cursor.execute("""
                insert into conversations (session_id, question, answer, timestamp)
                values (%s, %s, %s, now())
            """, (session_id, question, answer))
            # 获取更新后的对话历史
            history = self._fetch_recent_history(session_id)
            # 删除超过5轮的对话记录
            self.mysql_client.cursor.execute("""
                delete from conversations where session_id = %s and id not in (
                    select id from (
                        select id from conversations where session_id = %s order by timestamp desc limit %s
                    )as sub
                )
            """, (session_id, session_id, 5))
            # 提交数据库事务
            self.mysql_client.connection.commit()
            # 记录会话历史更新成功日志
            self.logger.info(f'会话历史更新成功：{session_id}')
            return history
        except pymysql.MySQLError as e:
            # 记录会话历史更新失败的错误日志
            self.logger.error(f'会话历史更新失败：{e}')
            # 回滚事务
            self.mysql_client.connection.rollback()
        except Exception as e:
            # 记录意外错误的日志
            self.logger.error(f'更新会话历史发生意外错误：{e}')
            # 回滚事务
            self.mysql_client.connection.rollback()
            # 抛出异常
            raise

    # todo 2.7清除会话历史：删除指定会话的所有记录
    def clear_session_history(self, session_id: str) -> bool:
        """
        函数作用：删除指定会话的所有记录
        :param session_id:会话唯一标识
        :return:True--》清除成功
        """
        try:
            # 删除指定的session_id的所有对话记录
            self.mysql_client.cursor.execute("""
                delete from conversations where session_id = %s
            """, (session_id,))
            # 提交数据库事务
            self.mysql_client.connection.commit()
            # 记录会话历史清除成功日志
            self.logger.info(f'会话历史清除成功：{session_id}')
            return True
        except pymysql.MySQLError as e:
            # 记录会话历史清除失败的错误日志
            self.logger.error(f'会话历史清除失败：{e}')
            # 回滚事务
            self.mysql_client.connection.rollback()
            return False

    # todo 2.8处理查询的核心方法：整合检索，大模型生成和对话历史，返回流式输出
    def query(self, query, source_filter=None, session_id=None):
        """
        查询集成系统，实现对话历史和流式输出
        """
        # 记录查询开始时间
        start_time = time.time()
        # 记录查询信息到日志
        self.logger.info(f'处理查询：{query}，会话id：{session_id}')
        # 获取会话历史，若无session_id 则返回空列表
        history = self.get_conversation_history(session_id) if session_id else []
        # 执行BM25搜索，获取答案和是否需要RAG标识
        answer, need_rag = self.bm25_search.search(query, threshold=0.85)
        if answer:
            # 如果找到可靠答案--》即：BM25阈值>0.85，记录答案到日志
            self.logger.info(f'MySQL答案：{answer}')
            if session_id:
                # 更新对话历史
                self.update_session_history(session_id, query, answer)
            # 计算处理时间
            process_time = time.time() - start_time
            self.logger.info(f'查询处理时间：{process_time}秒')
            # 返回答案，标记为完整
            yield answer, True
        elif not need_rag:
            # 如果需要RAG，记录回退信息到日志
            self.logger.info(f'无可靠MySQL答案，回退到RAG')
            # 初始化收集完整答案的字符串
            collected_answer = ""
            # 从RAG系统获取流式输出
            for token in self.rag_system.generate_answer(query, source_filter=source_filter, history=history):
                # 累计答案
                collected_answer += token
                # 返回当前答案片段，标记为非完整
                yield token, False
            if session_id:
                # 更新对话历史，存储完整答案
                self.update_session_history(session_id, query, collected_answer)
            # 计算处理时间
            process_time = time.time() - start_time
            self.logger.info(f'查询处理时间：{process_time}秒')
            # 返回空字符串，标记为结束
            yield "", True
        else:
            # 若无答案
            self.logger.info('未找到答案')
            # 计算处理时间
            process_time = time.time() - start_time
            self.logger.info(f'查询处理时间：{process_time}秒')
            yield '未找到答案', True


# todo 3.定义main函数：提供命令行交互界面，测试集成问答系统
def main():
    # 定义主函数，提供命令交互界面--》初始化 问答系统
    qa_system = IntegratedQASystem()
    # 生成唯一会话ID
    session_id = str(uuid.uuid4())
    # 打印欢迎信息
    print(f'欢迎使用集成问答系统！')
    # 打印会话ID
    print(f'会话ID：{session_id}')
    # 打印支持的学科
    print(f'支持的学科：{qa_system.config.VALID_SOURCES}')
    # 提示用户输入查询或退出
    print('输入你的问题，或输入"exit"来结束对话')
    try:
        while True:
            # 获取用户输入的问题
            query = input("\n请输入你的问题：").strip()
            if query.lower() == "exit":
                # 记录退出日志
                logger.info('退出系统')
                print('感谢使用！')
                break
            # 获取用户输入的学科过滤--》允许用户指定学科类别，无效则忽略
            source_filter = input(f'请输入学科类别：{"/".join(qa_system.config.VALID_SOURCES)}(直接回车跳过)').strip()
            if source_filter and source_filter not in qa_system.config.VALID_SOURCES:
                # 如果学科过滤无效，记录警告日志
                logger.warning(f'无效的学科过滤：{source_filter},将不进行过滤')
                # 设置学科过滤为空
                source_filter = None
            # 打印提示信息
            print(f'\n答案：', end="", flush=True)
            # 初始化累计答案的字符串
            answer = ''
            # 迭代query方法的生成器
            for token, is_complete in qa_system.query(query, source_filter=source_filter, session_id=session_id):
                if token:
                    # 仅当token非空时打印
                    print(token, end='', flush=True)
                    # 累计答案
                    answer += token
                if is_complete:
                    # 如果是完整答案或流结束，换行并退出循环
                    print()
                    break
            # 打印对话历史
            history = qa_system.get_conversation_history(session_id)
            print('最近对话历史')
            for idx, entry in enumerate(history, 1):
                # 按顺序打印历史记录
                print(f'{idx}. 问：{entry["question"]}\n答：{entry["answer"]}')
    except Exception as e:
        # 记录系统错误日志
        logger.error(f'系统错误：{e}')
        print(f'发生错误：{e}')
    finally:
        # 关闭MySQL连接
        qa_system.mysql_client.close()


# todo 4.程序主入口
if __name__ == '__main__':
    main()
