# MySQL数据库的基础操作

# 导入MySQL操作和数据处理依赖库
import pymysql
import pandas as pd
import sys, os

# todo 1.路径配置
# 获取当前文件所在目录的绝对路径，即：当前脚本所在的文件夹
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取当前目录的上一级目录
module_dir = os.path.dirname(current_dir)

# 获取项目根目录的绝对路径
project_dir = os.path.dirname(module_dir)

# 将项目根目录加入系统路径
sys.path.insert(0, project_dir)

# 导入自定义模块
from base import Config, logger


# todo 2.定义MySQL客户端类--》封装数据库连接，表操作，数据增差，连接关闭等功能
class MySQLClient:
    # todo 2.1初始化方法--》建立数据库连接，创建游标，处理连接异常
    def __init__(self):
        # 将全局logger赋值为类属性
        self.logger = logger
        try:
            # 建立MySQL数据库连接--》从config中读取编码
            self.connection = pymysql.connect(
                host=Config().MYSQL_HOST,
                user=Config().MYSQL_USER,
                password=Config().MYSQL_PASSWORD,
                database=Config().MYSQL_DATABASE,
            )
            # 创建游标对象--》用于执行SQL语句并返回结果
            self.cursor = self.connection.cursor()

            # 记录info日志--》确认Mysql连接成功
            self.logger.info("Mysql连接成功")

        except pymysql.MySQLError as e:
            self.logger.error(f"数据库连接失败：{e}")
            raise  # 抛出异常，不屏蔽失败，让调用者处理

    # todo 2.1.1连接探活--》连接被 MySQL 的 wait_timeout 掐掉后自动重连
    def _ensure_connection(self):
        """
        函数作用：每次执行 SQL 前探活，连接断了就自动重连
        解决：MySQL server has gone away (2006) 之后所有操作都变成 (0, '') 的问题
        """
        try:
            # ping 会在连接可用时发一个轻量包；reconnect=True 表示断开时自动重连
            self.connection.ping(reconnect=True)
        except Exception as e:
            # ping 失败或重连失败--》彻底重建连接和游标
            self.logger.warning(f"数据库连接已断开，正在重连：{e}")
            self.connection = pymysql.connect(
                host=Config().MYSQL_HOST,
                user=Config().MYSQL_USER,
                password=Config().MYSQL_PASSWORD,
                database=Config().MYSQL_DATABASE,
            )
            self.cursor = self.connection.cursor()
            self.logger.info("Mysql重连成功")

    # todo 2.2创建数据库表--》若jpbk表不存在则创建，定义表结构
    def create_table(self):
        # 创建表SQL语句，创建jpkb表，存储：学科知识问答数据
        create_table_query = """
        create table if not exists jpkb (
            id int auto_increment primary key,      # 自增主键
            subject_name varchar(20),               # 学科名称
            question varchar(1000),                 # 问题内容
            answer varchar(1000))                   # 答案内容
        """
        try:
            # 执行SQL前先探活，断了就重连
            self._ensure_connection()
            # 执行SQL语句--》创建表
            self.cursor.execute(create_table_query)
            # 提交事务--》确认创建表成功        MySQL默认事务要手动提交，确保表创建成功
            self.connection.commit()
            # 记录info日志--》确认创建表成功
            self.logger.info("表创建成功")
        except pymysql.MySQLError as e:
            self.logger.error(f"创建表失败：{e}")
            raise

    # todo 2.3从csv文件中批量插入数据到jpbk表中
    def insert_data(self, csv_file_path):
        try:
            # 执行SQL前先探活，断了就重连
            self._ensure_connection()
            # 读取csv文件--》获取数据
            data = pd.read_csv(csv_file_path)
            # 打印数据
            # print(data.head())

            # 循环遍历csv每一行数据--》执行插入操作
            for _, row in data.iterrows():
                # 定义插入SQL语句，使用%s占位符，避免SQL注入攻击问题
                insert_query = 'insert into jpkb(subject_name, question, answer) values (%s, %s, %s)'
                # 执行插入语句--》将csv行数据(学科名称，问题，答案)传入占位符
                self.cursor.execute(insert_query, (row['学科名称'], row['问题'], row['答案']))
            # 提交事务--》确认所有插入操作批量生效
            self.connection.commit()
            self.logger.info("数据插入成功")

        except Exception as e:
            # 输出错误信息
            self.logger.error(f"数据插入失败：{e}")
            # 事务回滚：把数据库连接恢复到事务开始前的状态--》类似linux的快照
            # 注意：连接已死时 rollback() 自身会再抛一次异常，必须包住
            try:
                self.connection.rollback()
            except Exception:
                pass
            # 抛出异常，不屏蔽失败，让调用者处理
            raise

    # todo 2.4从jpkb表中获取所有问题，返回问题列表
    def fetch_questions(self):
        try:
            # 执行SQL前先探活，断了就重连
            self._ensure_connection()
            # 执行SQL查询
            self.cursor.execute("select question from jpkb")
            # 获取查询结果，fetch_all()返回所有查询结果
            result = self.cursor.fetchall()
            # 记录日志
            self.logger.info("问题列表获取成功")
            return result
        except pymysql.MySQLError as e:
            # 记录错误日志
            self.logger.error(f"获取问题列表失败：{e}")
            return []

    # todo 2.5根据指定问题从jpkb表中获取对应的答案
    def fetch_answer(self, question):
        try:
            # 执行SQL前先探活，断了就重连
            self._ensure_connection()
            # 执行SQL查询
            self.cursor.execute("select answer from jpkb where question = %s", (question,))
            # 获取查询结果
            result = self.cursor.fetchone()
            # 记录日志
            self.logger.info("答案获取成功")
            # 处理结果，有匹配数据则返回答案（取元组的第一个元素），无匹配返回None

            return result[0] if result else None
        except pymysql.MySQLError as e:
            # 记录错误日志
            self.logger.error(f"获取答案失败：{e}")
            return None

    # todo 2.6关闭数据库连接-->释放资源，避免连接泄露
    def close(self):
        try:
            # 关闭连接
            self.connection.close()
            self.logger.info("数据库连接关闭成功")
        except pymysql.MySQLError as e:
            # 记录错误日志
            self.logger.error(f"数据库连接关闭失败：{e}")


# todo 3.测试代码
if __name__ == '__main__':
    # 实例化MySQLClient客户端类--》创建数据库连接
    mysql_client = MySQLClient()

    # 功能性测试代码
    # mysql_client.create_table()

    # 批量插入数据
    # mysql_client.insert_data('../data/JP学科知识问答.csv')

    # 获取所有的问题--》返回问题列表
    # questions = mysql_client.fetch_questions()

    # 根据问题获取答案
    anwer = mysql_client.fetch_answer('redis存储类型里面提到的“string类型是二进制安全的')
    print(anwer)

    # 关闭数据库连接
    mysql_client.close()
