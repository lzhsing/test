# 自定义的日志工具类

# 导包
import logging
import os
from config import Config

# todo 1.计算日志文件的最终保存路径
module_path = os.path.abspath(__file__)
# print(module_path)  # D:\Projects\edu_rag_code\day03\logging_module\utils\logger.py

# 获取当前脚本所在的路径
base_path = os.path.dirname(module_path)
# print(base_path)  # D:\Projects\edu_rag_code\day03\logging_module\utils

# 获取当前脚本所在目录的父目录的路径
current_path = os.path.dirname(base_path)
# print(current_path)  # D:\Projects\edu_rag_code\day03\logging_module

# 拼接日志文件的完整路径       上级目录+logs文件夹+app.log文件
# log_file = os.path.join(current_path, 'logs/app.log')
log_file = os.path.join(current_path, Config().LOG_FILE)
# print(log_file)  # D:\Projects\edu_rag_code\day03\logging_module\logs/app.log


# todo 2.封装日志配置函数--》可重复使用的双输出日志
def setup_logger( log_file=log_file):
    """
    函数功能：创建并返回一个日志记录器，支持日志同时输出到控制台和文件，且避免重复添加处理器
    :param name:日志记录器的名称--》用于区分不同模块的日志
    :param log_file:日志文件的保存路径
    :return:日志记录器
    """
    # 确保日志目录存在
    # 参1：获取日志的所在目录      参2：若目录已存在，不报错，若不存在则创建
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # 创建日志记录器对象--》统一管理日志处理器，并设置日志级别
    logger = logging.getLogger('Edu_RAG')
    logger.setLevel(level=logging.INFO)  # 后续处理器可设置更高级别，实现：记录器收全，处理器筛选

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level=logging.INFO)
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(level=logging.INFO)

    # 创建日志输出格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 为两个处理器分别绑定日志格式
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # 为日志记录器添加两个处理器，判断处理器是否存在，避免重复调用函数导致重复添加
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    # 返回日志信息
    return logger


# todo 3.初始化日志记录器--》项目启动时自动调用，其他模块导入次logger日志器即可直接使用
logger = setup_logger()
