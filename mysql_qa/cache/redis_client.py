# Redis缓存技术

# 导包
import redis
import json
import os, sys

# todo 0.路径相关
# 获取当前文件所在目录的绝对目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取当前目录的上一级目录
parent_dir = os.path.dirname(current_dir)

# 获取项目根目录的绝对路径
project_root_dir = os.path.dirname(parent_dir)

# 将项目根目录添加到系统路径中
sys.path.insert(0, project_root_dir)

# 导入自定义配置和日志模块
from base import Config, logger


# todo 1.定义Redis客户端类--》封装Redis连接，数据存储，数据获取，答案查询等功能
class RedisClient:
    # todo 1.1初始化方法，获取Redis连接，初始化日志，处理连接异常
    def __init__(self):
        # 获取日志记录器
        self.logger = logger
        try:
            # 建立Redis连接
            self.client = redis.StrictRedis(
                host=Config().REDIS_HOST,  # redis服务器主机地址
                port=Config().REDIS_PORT,
                password=Config().REDIS_PASSWORD,
                db=Config().REDIS_DATABASE,
                decode_responses=True  # 自动将返回的字节数据转成字符串（避免手动解码）
            )
        except redis.RedisError as e:
            self.logger.error(f'redis连接异常：{e}')
            raise  # 抛出异常

        # 记录info日志
        self.logger.info("Redis连接成功")

    # todo 1.2存储数据到Redis-->将python对象序列化为json字符串存储
    def set_data(self, key, value):
        try:
            # 存储数据
            self.client.set(key, json.dumps(value, ensure_ascii=False))  # ensure_ascii=False保证中文等非ASCII编码正常显示
            self.logger.info(f'redis存储数据成功：{key}')
        except redis.RedisError as e:
            self.logger.error(f'redis存储数据异常：{e}')

    # todo 1.3从Redis中获取数据--》将json字符串反序列化为python对象
    def get_data(self, key):
        try:
            # 获取数据
            data = self.client.get(key)
            # 处理结果--》将json字符串反序列化为python对象
            return json.loads(data) if data else None
        except redis.RedisError as e:
            self.logger.error(f'redis获取数据异常：{e}')
            return None

    # todo 1.4根据查询内容从Redis获取缓存答案--》键格式固定为："answer:{确认样}"
    def get_answer(self, query):
        try:
            # 构建键名
            answer = self.client.get(f'answer:{query}')
            # 若存在缓存答案，记录info日志，并返回内容
            if answer:
                self.logger.info(f'redis获取答案成功：{query}')
                return answer
            # 到这表示不存在缓存答案
            return None
        except redis.RedisError as e:
            self.logger.error(f'redis获取答案异常：{e}')
            return None


# todo 2.测试代码
if __name__ == '__main__':
    # 实例化Redis客户端
    redis_client = RedisClient()

    # 获取redis中所有的键
    # print(redis_client.client.keys("*"))

    # 测试get_data
    print(redis_client.get_data('用上下文管理器实现函数运行时间的计算？'))

    # 测试get_answer
    # print(redis_client.get_answer('用上下文管理器实现函数运行时间的计算'))