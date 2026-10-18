# 文本预处理

import jieba
from base import logger


# todo 1.定义文本预处理函数
def preprocess_text(text):
    # 预处理文本
    logger.info("开始预处理文本")
    try:
        # 分词并转为小写
        return jieba.lcut(text.lower())
    except AttributeError as e:
        logger.error(f"预处理文本异常：{e}")
        return []


# todo 2.程序主入口
if __name__ == '__main__':
    print(preprocess_text('主打歌的撒给大家仨'))
