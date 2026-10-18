# 导包
import os           # 用于处理文件路径
import sys          # 用于管理python解释器的系统路径，确保自定义模块能被正确导入

# 获取当前文件所在目录的绝对路径，即：当前脚本所在的文件夹
module_dir = os.path.dirname(os.path.abspath(__file__))

# 获取项目根目录的绝对路径
# 用途：将项目根目录加入系统路径，确保跨目录导入自定义模块
project_path = os.path.dirname(module_dir)

# 将模块目录加入系统路径
# 若模块目录不在sys.path中，插入到列表首位，确保优先加载当前模块目录下的文件，避免：同名模块冲突
if module_dir not in sys.path:
    sys.path.insert(0,module_dir)

# 将项目根目录加入系统路径
if project_path not in sys.path:
    sys.path.insert(0,project_path)

# 调用并打印
# print(sys.path)

# 导入自定义模块
from config import Config               # 导入配置解析类，用于：读取项目配置
from logger import logger               # 导入初始化好的日志实例，用于：记录项目日志

