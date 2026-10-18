# 配置文件管理

# 导包
import configparser  # 导入配置文件解析库，解析INI格式的配置文件
import os


# todo 1.配置文件路径计算
current_path = os.path.abspath(__file__)
# print(current_path) # D:\Projects\/edu_rag_code\integrated_qa_system\base\config.py

# 获取当前文件所在目录的绝对路径
current_dir_path = os.path.dirname(current_path)
# print(current_dir_path)   # D:\Projects\edu_rag_code\integrated_qa_system\base

# 获取项目根目录的路径
project_path = os.path.dirname(current_dir_path)
# print(project_path)     # D:\Projects\edu_rag_code\integrated_qa_system

# 拼接配置文件
config_path = os.path.join(project_path, 'config.ini')
# print(config_path)


# todo 2.配置解析类--》封装配置文件读取逻辑，提供统一的配置参数访问接口
class Config:
    # todo 2.1初始化方法：读取配置文件并解析各服务的配置参数
    def __init__(self, config_file=config_path):
        # # 1.创建配置文件解析器实例
        # self.config = configparser.ConfigParser()
        # # 2.读取配置文件--》根据传入的路径加载ini文件中的内容
        # self.config.read(config_file,encoding='utf-8')
        #
        # # 3.解析并存储各服务的配置参数
        # # 3.1解析MySQL数据库配置参数
        # self.MYSQL_HOST = self.config.get('mysql','host',fallback='localhost')
        # self.MYSQL_USER = self.config.get('mysql','user',fallback='root')
        # self.MYSQL_PASSWORD = self.config.get('mysql','password',fallback='123456')
        # self.MYSQL_DATABASE = self.config.get('mysql','database',fallback='subject_kg')
        #
        # # 3.2解析Redis数据库配置参数
        # self.REDIS_HOST = self.config.get('redis','host',fallback='localhost')
        # self.REDIS_PORT = self.config.getint('redis','port',fallback=6379)
        # self.REDIS_PASSWORD = self.config.get('redis','password',fallback='1234')
        # self.REDIS_DATABASE = self.config.getint('redis','database',fallback=0)
        #
        # # 3.3解析日志配置
        # # 日志文件路径：从[logger]节点的"log_file"属性中获取，默认值为logs/app.log
        # self.LOG_FILE = self.config.get('logger','log_file',fallback='logs/app.log')

        # 创建配置解析器，启用插拔功能
        self.config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        # 如果没有提供配置文件路径，则使用默认路径
        self.PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.LOG_DIR = os.path.join(self.PROJECT_ROOT, 'logs')
        self.DATA_DIR = os.path.join(self.PROJECT_ROOT, 'rag_qa/data')
        self.MODELS_DIR = os.path.join(self.PROJECT_ROOT, 'rag_qa/models')
        self.EDU_DOCUMENT_LOADERS_DIR = os.path.join(self.PROJECT_ROOT, 'rag_qa/edu_document_loaders')
        if not config_file:
            config_file = os.path.join(self.PROJECT_ROOT, 'config.ini')
        # 读取配置文件的信息
        self.config.read(config_file, encoding='utf-8')

        # MySQL配置
        self.MYSQL_HOST = os.getenv('MYSQL_HOST', self.config.get('mysql', 'host', fallback='localhost'))
        self.MYSQL_USER = os.getenv('MYSQL_USER', self.config.get('mysql', 'user', fallback='root'))
        self.MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', self.config.get('mysql', 'password', fallback='123456'))
        self.MYSQL_DATABASE = os.getenv('MYSQL_DATABASE', self.config.get('mysql', 'database', fallback='subject_kg'))

        # Redis配置
        self.REDIS_HOST = os.getenv('REDIS_HOST', self.config.get('redis', 'host', fallback='localhost'))
        self.REDIS_PORT = int(os.getenv('REDIS_PORT', self.config.getint('redis', 'port', fallback=6379)))
        self.REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', self.config.get('redis', 'password', fallback='1234'))
        self.REDIS_DATABASE = int(os.getenv('REDIS_DATABASE', self.config.get('redis', 'database', fallback=0)))

        # Milvus配置
        self.MILVUS_HOST = os.getenv('MILVUS_HOST', self.config.get('milvus', 'host', fallback='localhost'))
        self.MILVUS_PORT = os.getenv('MILVUS_PORT', self.config.get('milvus', 'port', fallback=19530))
        self.MILVUS_DATABASE_NAME = os.getenv('MILVUS_DATABASE_NAME',
                                              self.config.get('milvus', 'database_name', fallback='itcast'))
        self.MILVUS_COLLECTION_NAME = os.getenv('MILVUS_COLLECTION_NAME',
                                                self.config.get('milvus', 'collection_name', fallback='edurag'))

        # LLM配置
        self.MODEL_NAME = os.getenv('MODEL_NAME', self.config.get('llm', 'model_name', fallback='qwen3.8-27b'))
        self.DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY', self.config.get('llm', 'dashscope_api_key',
                                                                                fallback='5fe74f7e3a3e47d49cc08b74100e79d1'))
        self.DASHSCOPE_BASE_URL = os.getenv('DASHSCOPE_BASE_URL', self.config.get('llm', 'dashscope_base_url',
                                                                                       fallback='https://dashscope.aliyuncs.com/compatible-mode/v1'))

        # 检索参数
        # 父块大小
        self.PARENT_CHUNK_SIZE = self.config.getint('retrieval', 'parent_chunk_size', fallback=1200)
        # 子块大小
        self.CHILD_CHUNK_SIZE = self.config.getint('retrieval', 'child_chunk_size', fallback=300)
        # 块重叠大小
        self.CHUNK_OVERLAP = self.config.getint('retrieval', 'chunk_overlap', fallback=50)
        # 检索返回数量
        self.RETRIEVAL_K = self.config.getint('retrieval', 'retrieval_k', fallback=5)
        # 最终候选数量
        self.CANDIDATE_M = self.config.getint('retrieval', 'candidate_m', fallback=2)

        # 应用配置
        # 有效来源列表
        self.VALID_SOURCES = eval(
            self.config.get('app', 'valid_sources', fallback='["ai", "java", "test", "ops", "bigdata"]'))
        # 客服电话
        self.CUSTOMER_SERVICE_PHONE = self.config.get('app', 'customer_service_phone', fallback='12345678')

        # 日志文件路径
        self.LOG_FILE = os.path.join(self.LOG_DIR, 'app.log')


# todo 3.主程序入口
if __name__ == '__main__':
    # 手动指定配置文件路径--》用于测试，实际使用可忽略
    # config_file = r'D:\Projects\edu_rag_code\integrated_qa_system\config.ini'
    config_file = '../config.ini'

    # 实例化Config类--》加载并解析配置文件
    conf = Config(config_file)
    # 测试配置参数
    print(conf.MYSQL_HOST, conf.MYSQL_USER, conf.MYSQL_PASSWORD, conf.MYSQL_DATABASE)
