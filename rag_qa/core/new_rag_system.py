# RAG系统核心逻辑

# 导包
import os, sys
import time
from base.config import Config
from base.logger import logger
from rag_qa.core.prompts import RAGPrompts  # 导入RAG相关提示模板--》定义大模型的输入和输出格式
from rag_qa.core.query_classifier import QueryClassifier  # 导入查询分类器
from rag_qa.core.strategy_selector import StrategySelector  # 导入策略选择器
from rag_qa.core.vector_store import VectorStore  # 导入向量数据库

# 优化1
from transformers import BertTokenizer, BertModel
local_model_path = r'D:\Projects\edu_rag_code\intergrated_qa_system\rag_qa\models\bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(local_model_path)
model = BertModel.from_pretrained(local_model_path)
from openai import OpenAI

# todo 1.配置项目路径-->确保程序能正确导入其他目录的模块
# 获取当前文件目录的绝对路径
current_path = os.path.dirname(os.path.abspath(__file__))
# 获取上级目录的路径
rag_qa_path = os.path.dirname(current_path)
# 将rag_qa_path添加到sys.path
sys.path.insert(0, rag_qa_path)
# 获取项目根目录路径
project_path = os.path.dirname(rag_qa_path)
# 将project_path添加到sys.path
sys.path.insert(0, project_path)


# todo 2.定义RAGSystem类--》封装RAG系统的整体流程，即：查询分类--》策略选择--》文档检索--》答案生成
class RAGSystem:
    # todo 2.1初始化方法--》创建RAG系统所需的核心组件--》向量库，大模型，分类器...
    def __init__(self, vector_store, llm):
        """
        初始化RAG系统
        :param vector_store:向量数据库对象--》用于存储和检索文档，提供相似度搜索功能
        :param llm:大语言模型调用函数--》接受提示词文本，返回模型生成的答案
        """
        # 保存向量数据库对象
        self.vector_store = vector_store
        # 保存大语言模型调用函数
        self.llm = llm
        # 加载RAG提示模板--》定义生成答案时的固定格式
        self.rag_prompts = RAGPrompts.rag_prompt()
        # 初始化查询分类器--》用于判断用户查询类型（通用知识还是专业咨询）
        # 拼接分类器模型路径
        classifier_model_path = os.path.join(rag_qa_path, "models", "bert_query_classifier")
        # 创建查询分类器实例
        self.query_classifier = QueryClassifier(classifier_model_path)
        # 初始化策略选择器--》用于专业咨询策略选择
        self.strategy_selector = StrategySelector()

    # todo 2.2定义私有方法，使用假设文档进行查询（HYDE）-->生成假设答案，用假设答案检索相关文档
    def _retrieve_with_hyde(self, query, source_filter=None):
        """
        针对于抽象/开放性查询，先生成假设性答案，再用假设答案检索相关文档--》解决抽象查询直接检索关键词匹配度低的问题
        :param query:用户的查询问题
        :return:list[Document]，每个Document对象的page_content属性为文档文本，用于后续生成答案的上下文
        """
        logger.info(f'使用HyDE策略进行检索（查询问题：{query}）')
        # 获取假设问题生成的Prompt模板
        hyde_prompt_template = RAGPrompts.hyde_prompt()
        # 调用大语言模型生成假设问题
        try:
            hyde_answer = self.llm(hyde_prompt_template.format(query=query).strip())
            logger.info(f'HyDE假设问题生成结果：{hyde_answer}')
            # 使用假设答案进行检索，并返回检索结果
            # 注意：HyDE通常只用于生成检索向量，不一定需要rerank这一步，这里复用了
            return self.vector_store.hybrid_retrieve_and_reorder(
                hyde_answer,
                k=Config().RETRIEVAL_K,
                source_filter=source_filter
            )
        except Exception as e:
            logger.error(f'HyDE策略执行失败：{e}')
            return []

    # todo 2.3定义私有方法，使用子查询进行检索--》复杂查询拆分为子查询，分别检索，最后合并结果
    def _retrieve_with_subquery(self, query, source_filter=None):
        """
        针对于多实体/多方面的复杂查询，拆分为多个简单的子查询，分别检索后合并去重--》解决多维度查询检索不全面的问题
        :param query:用户的查询问题
        :return:list[Document]，每个Document对象的page_content属性为文档文本，用于后续生成答案的上下文
        """
        logger.info(f'使用SubQuery策略进行检索（查询问题：{query}）')
        # 获取子查询生成的Prompt模板
        subquery_prompt_template = RAGPrompts.subquery_prompt()
        try:
            # 调用大语言模型生成子查询
            subqueries_answer = self.llm(subquery_prompt_template.format(query=query).strip())
            subqueries = [q.strip() for q in subqueries_answer.split('\n') if q.strip()]
            logger.info(f'子查询生成结果：{subqueries}')
            if not subqueries:
                logger.warning("没有生成有效的子查询")
                return []

            # 初始化空列表，用于存储所有子查询的检索结果
            all_docs = []
            # 遍历每个子查询
            for sub_q in subqueries:
                # 使用子查询进行检索，并将结果添加到列表中
                # 这里对每个子查询都执行了hybird_search+rerank,
                docs = self.vector_store.hybrid_retrieve_and_reorder(
                    sub_q,
                    source_filter=source_filter,
                    k=Config().RETRIEVAL_K
                )
                all_docs.extend(docs)
                logger.info(f'子查询“{sub_q}”检索到{len(docs)}个文档')

            # 对所有检索结果进行去重（基于对象内存地址，如果Document内容相同但对象不同则无法去重）
            # 更可靠的去重方式：基于文档内容或ID
            unique_docs = list({doc.page_content: doc for doc in all_docs}.values())

            logger.info(f'所有子查询检索结果去重后共得到{len(unique_docs)}个文档')
            # 返回去重后的文档，限制数量
            return unique_docs[:Config().CANDIDATE_M]

        except Exception as e:
            logger.error(f'子查询策略执行失败：{e}')
            return []

    # todo 2.4定义私有方法，使用回溯问题进行检索--》将复杂查询简化为基础问题后再检索
    def _retrieve_with_backtracking(self, query, source_filter=None):
        logger.info(f'使用Backtracking策略进行检索（查询问题：{query}）')
        backtracking_prompt_template = RAGPrompts.backtracking_prompt()
        try:
            # 调用大语言模型生成回溯问题
            backtracking_answer = self.llm(backtracking_prompt_template.format(query=query).strip())
            logger.info(f'Backtracking问题生成结果：{backtracking_answer}')
            # 使用回溯问题进行检索，并返回检索结果
            return self.vector_store.hybrid_retrieve_and_reorder(
                backtracking_answer,
                k=Config().RETRIEVAL_K,
                source_filter=source_filter
            )
        except Exception as e:
            logger.error(f'Backtracking策略执行失败：{e}')
            return []

    # todo 2.5根据检索策略检索文档，用合并/筛选最终上下文对象
    def retrieve_and_merge(self, query, source_filter=None, strategy=None):
        """
        统一入口：根据指定检索策略（或自动选择）调用相应的检索方法，筛选最终的上下文文档
        :param query:用户的原始查询文本--》str，传递给策略选择器和检索方法
        :param source_filter:检索来源过滤条件
        :param strategy:选择的检索策略--》str，如"hyde"、"subquery"、"backtracking"
        :return:list[Document]
        """
        # 如果未指定检索策略，则使用策略选择器选择
        if not strategy:
            strategy = self.strategy_selector.select_strategy(query)
        # 根据策略调用对应的检索方法--》获取候选文档列表
        ranker_chunks = []
        if strategy == "假设问题检索":
            ranker_chunks = self._retrieve_with_hyde(query)
        elif strategy == "子查询检索":
            ranker_chunks = self._retrieve_with_subquery(query)
        elif strategy == "回溯问题检索":
            ranker_chunks = self._retrieve_with_backtracking(query)
        else:
            logger.info(f'使用默认检索策略（查询问题：{query}）')
            ranker_chunks = self.vector_store.hybrid_retrieve_and_reorder(
                query,
                k=Config().RETRIEVAL_K,
                source_filter=source_filter
            )
        # 选择最终上下文文档，截取前CANDIDATE_M个文档（控制上下文长度，避免LLM输入超限）
        logger.info(f'策略：{strategy}，检索到{len(ranker_chunks)}个文档')
        final_context_docs = ranker_chunks[:Config().CANDIDATE_M]
        logger.info(f'最终选择{len(final_context_docs)}个文档作为上下文')

        # 返回最终上下文文档
        return final_context_docs

    # 优化2
    # todo 2.6定义方法，生成答案
    def generate_answer(self, query, source_filter=None,history=None):
        """
        根据查询类型选择直接用LLM或者执行RAG流程，支持流式输出
        :param query: 用户查询文本
        :param source_filter: 检索来源过滤条件,仅对‘专业咨询’策略有效
        :param history: 对话历史，包含字典{'question': '用户问题', 'answer': '模型答案'}
        :return: 生成器，逐段返回答案文本
        """
        # 记录查询开始时间，用于计算整个查询的处理耗时
        start_time = time.time()
        logger.info(f'开始处理查询：{query},学科过滤：{source_filter}')

        # 1. 验证并处理对话历史, 确保格式正确, 只保留最近5轮
        if history is not None and not isinstance(history, list):
            logger.warning(f"无效的历史格式: {type(history)}，忽略历史")
            history = []
        elif history:
            history = history[-5:]  # 限制最多5轮 -> 即: 限制历史长度, 避免上下文过程.
            for h in history:
                if not (isinstance(h, dict) and "question" in h and "answer" in h):
                    logger.warning(f"无效的历史条目: {h}，忽略历史")
                    history = []
                    break

        # 2. 构造历史上下文, 格式化历史记录为字符串.
        history_context = ""
        if history:
            history_context = "\n".join(
                [f"Q: {h['question']}\nA: {h['answer']}" for h in history]
            )
            logger.info(f"使用对话历史: {history_context[:100]}...")

        # 3. 判断查询类型 -> 通过分类器区分通用知识 和 专业咨询.
        query_category = self.query_classifier.predict_category(query)
        logger.info(f"查询分类结果：{query_category} (查询: '{query}')")

        #  4. 如果查询属于“通用知识”类别，则直接使用 LLM 回答, 专业咨询则进行 RAG 检索.
        if query_category == "通用知识":
            logger.info("查询为通用知识，直接调用 LLM")
            prompt_input = self.rag_prompts.format(
                context="", history=history_context, question=query, phone=Config().CUSTOMER_SERVICE_PHONE
            )  # 不使用上下文
            try:
                answer = self.llm(prompt_input)
            except Exception as e:
                logger.error(f"直接调用 LLM 失败: {e}")
                answer = f"抱歉，处理您的通用知识问题时出错。请联系人工客服：{Config().CUSTOMER_SERVICE_PHONE}"
            processing_time = time.time() - start_time
            logger.info(
                f"通用知识查询处理完成 (耗时: {processing_time:.2f}s, 查询: '{query}')"
            )
            return answer

        #   否则，进行 RAG 检索并生成答案
        logger.info("查询为专业咨询，执行 RAG 流程")
        # 5. 选择检索策略
        strategy = self.strategy_selector.select_strategy(query)

        # 6. 检索相关文档
        context_docs = self.retrieve_and_merge(
            query, source_filter=source_filter, strategy=strategy
        )  # 传递 strategy

        #  7. 准备上下文
        if context_docs:
            context = "\n\n".join([doc.page_content for doc in context_docs])  # 使用换行符分隔文档
            logger.info(f"构建上下文完成，包含 {len(context_docs)} 个文档块")
            # logger.debug(f"上下文内容:\n{context[:500]}...") # Debug 日志可以打印部分上下文
        else:
            context = ""
            logger.info("未检索到相关文档，上下文为空")

        #  8. 构造 Prompt，调用大语言模型生成答案
        # 构造提示
        prompt_input = self.rag_prompts.format(
            context=context,
            history=history_context,
            question=query,
            phone=Config().CUSTOMER_SERVICE_PHONE
        )
        logger.debug(f"最终生成的 Prompt:\n{prompt_input}") # Debug 日志

        # 9. 调用大模型生成流式答案.
        try:
            answer = self.llm(prompt_input)
        except Exception as e:
            logger.error(f"调用 LLM 生成最终答案失败: {e}")
            answer = f"抱歉，处理您的专业咨询问题时出错。请联系人工客服：{Config().CUSTOMER_SERVICE_PHONE}"

        # 记录查询处理完成的日志
        processing_time = time.time() - start_time
        logger.info(f"查询处理完成 (耗时: {processing_time:.2f}s, 查询: '{query}')")
        return answer


# 测试代码
if __name__ == '__main__':
    # 实例化向量数据库
    vector_store = VectorStore()
    # 定义大语言模型调用函数
    llm = StrategySelector().call_dashscope

    # 创建RAGSystem实例
    rag_system = RAGSystem(vector_store, llm)

    # 测试生成答案
    # answer = rag_system.generate_answer("AI学科的课程大纲是什么",source_filter='ai')
    # answer = rag_system.generate_answer("AI学科的课程大纲的内容是什么",source_filter='bigdata')
    # answer = rag_system.generate_answer("请介绍一下Python Web开发课程的项目案例。",source_filter='ai')
    answer = rag_system.generate_answer("生成一个长度为10的随机密码，包含大小写字母和数字。", source_filter='ai')
    # 打印模型生成的答案--》实际部署时可改为：返回给前端或者存储
    print(answer)
