# 向量存储和检索
# 核心流程：文档向量生成（BGE-M3）--》向量入库（Milvus）--》混合检索（稠密+稀疏向量）--》结果去重--》重排序--》返回精准文档


# 模型设备相关: 判断GPU是否可用, 为模型选择运行设备.
import torch.cuda
# 导入 BGE-M3 嵌入函数，用于生成文档和查询的向量表示
from milvus_model.hybrid import BGEM3EmbeddingFunction
# 导入 Milvus 相关类，用于操作向量数据库, 例如: 客户端, 数据类型, 检索请求, 排序器
from pymilvus import MilvusClient, DataType, AnnSearchRequest, WeightedRanker
# 导入 Document 类，用于创建文档对象, 即: 统一文档数据格式(含内容 + 元数据)
from langchain_core.documents import Document
# 导入 CrossEncoder，用于重排序和 NLI 判断 -> 优化检索结果的相关性排序.
from sentence_transformers import CrossEncoder
# 导入 hashlib 模块，用于生成唯一 ID 的哈希值 -> 作为Milvus的主键.
import hashlib
import sys, os
from rag_qa.core.document_processor import process_documents

# from document_processor import *        # 导入自定义的文档加载和切分函数.

# todo 1.路径配置模块，定位当前脚本，RAG核心目录，项目根目录，确保自定义模块可导入
local_path = os.path.abspath(os.path.dirname(__file__))
# print("local_path:", local_path)

rag_qa_path = os.path.abspath(os.path.dirname(local_path))
# print("rag_qa_path:", rag_qa_path)

project_root = os.path.dirname(rag_qa_path)
# print("project_root:", project_root)

sys.path.insert(0, project_root)

from base.logger import logger
from base.config import Config

# todo 2.初始化全局配置--》实例化配置对象
conf = Config()


# todo 3.定义VectorStore类：封装向量库的核心操作--》集合管理，文档入库，混合检索，结果处理
class VectorStore:
    # todo 3.1初始化方法--》配置向量库连接，加载模型，初始化客户端
    def __init__(
            self,
            collection_name=conf.MILVUS_COLLECTION_NAME,
            host=conf.MILVUS_HOST,
            port=conf.MILVUS_PORT,
            database=conf.MILVUS_DATABASE_NAME
    ):
        # 存储Milvus的核心设置
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.database = database

        # 初始化日志实例
        self.logger = logger

        # 选择模型运行设备--》优先使用GPU，无GPU则使用CPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info(f'使用设备: {self.device}')

        # 初始化BGE_Reranker重排序模型--》优化检索结果相关性排序
        # 拼接重排序模型的本地路径，即：\rag_qa\models\bge-reranker-large
        reranker_model_path = os.path.join(rag_qa_path, 'models', 'bge-reranker-large')
        # 加载模型--》指定运行设备，模型用于计算‘查询问题‘和‘文档内容‘的相似度得分
        # low_cpu_mem_usage=True：边读盘边装权重，加载峰值从约 9GB 降到约 3.4GB，
        # 避免内存不足时 torch 在分配参数处踩空指针，Windows 报 0xC0000005(access violation)
        self.reranker = CrossEncoder(
            reranker_model_path,
            device=self.device,
            automodel_args={"low_cpu_mem_usage": True},
        )

        # 初始化BGE_M3嵌入函数--》生成文档和查询的向量表示
        # 拼接嵌入模型的本地路径，即：\rag_qa\models\bge-m3
        m3_path = os.path.join(rag_qa_path, 'models', 'bge-m3')
        # 加载模型--》模型用于生成文档和查询的向量表示
        self.embedder_function = BGEM3EmbeddingFunction(
            model_name_or_path=m3_path,  # 模型本地路径
            device=self.device,
            use_fp16=(self.device == 'cuda')  # GPU时启用半精度计算
        )

        # 获取BGE-M3稠密向量表示：固定输出1024维稠密向量
        self.dense_dim = self.embedder_function.dim['dense']
        # print(f'BGE-M3稠密向量表示维度: {self.dense_dim}')

        # 初始化Milvus客户端，建立向量数据库连接
        # uri='http://主机地址:端口号',db_name='数据库名'
        self.client = MilvusClient(uri=f'http://{self.host}:{self.port}', db_name=self.database)

        # 调用‘私有’方法--》创建新集合（若不存在）或加载已有集合（如存在）
        self._create_or_load_collection()

    # todo 3.2私有方法--》创建新集合（若不存在）或加载已有集合（如存在）
    def _create_or_load_collection(self):
        # 检查指定集合是否已存在(避免重复创建）
        if not self.client.has_collection(self.collection_name):
            # 创建集合Schema，禁用自动ID（手动用文档哈希值作为主键），启动动态字段
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
            # 添加ID字段，作为主键，VARCHAR类型，最大长度100
            schema.add_field(field_name='id', datatype=DataType.VARCHAR, max_length=100, is_primary=True)
            # 添加文本字段，VARCHAR类型，长度65535
            schema.add_field(field_name='text', datatype=DataType.VARCHAR, max_length=65535)
            # 添加稠密向量字段，FLOAT_VECTOR类型，维度由嵌入函数指定
            schema.add_field(field_name='dense_vector', datatype=DataType.FLOAT_VECTOR, dim=self.dense_dim)
            # 添加稀疏向量字段，SPARSE_FLOAT_VECTOR类型
            schema.add_field(field_name='sparse_vector', datatype=DataType.SPARSE_FLOAT_VECTOR)
            # 添加父块ID字段，VARCHAR类型，最大长度100
            schema.add_field(field_name='parent_id', datatype=DataType.VARCHAR, max_length=100)
            # 添加父块内容字段，VARCHAR类型，长度65535
            schema.add_field(field_name='parent_content', datatype=DataType.VARCHAR, max_length=65535)
            # 添加学科类别字段，VARCHAR类型，最大长度50
            schema.add_field(field_name='source', datatype=DataType.VARCHAR, max_length=50)
            # 添加时间戳字段，VARCHAR类型，最大长度50
            schema.add_field(field_name='timestamp', datatype=DataType.VARCHAR, max_length=50)

            # 创建索引参数对象
            index_params = self.client.prepare_index_params()
            # 为稠密向量字段添加IVF_FLAT索引，度量类型为内积（IP）
            index_params.add_index(
                field_name='dense_vector',
                index_name='dense_index',
                index_type='IVF_FLAT',
                metric_type='IP',
                params={'nlist': 128}  # IVF_FLAT索引参数，nlist为128
            )
            # 为稀疏向量添加SPARSE_INVERTED_INDEX索引，度量类型为内积（IP）
            index_params.add_index(
                field_name='sparse_vector',
                index_name='sparse_index',
                index_type='SPARSE_INVERTED_INDEX',
                metric_type='IP',
                params={'drop_ratio_build': 0.2}  # 构建索引时，丢弃20%的低权重值（减少存储，噪声，精度影响较小）
            )

            # 创建Milvus集合，应用定义的Schema和索引参数
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params
            )
            # 记录创建集合日志
            self.logger.info(f'集合 {self.collection_name} 创建成功')
        # 如果集合已经存在
        else:
            # 记录加载集合的日志
            self.logger.info(f'集合 {self.collection_name} 已存在，加载成功')
        # 将集合加载到内存，确保可立即查询
        self.client.load_collection(self.collection_name)

    # todo 3.3定义函数--》将文档（子块）转换为向量并插入Milvus
    def add_documents(self, documents):
        # 提取所有文档内容列表
        texts = [doc.page_content for doc in documents]

        # 使用BGE-M3模型生成文档向量
        # 输入：文本列表（texts）
        # 输出：字典，含‘dense’和‘sparse’向量
        embeddings = self.embedder_function(texts)

        # 初始化空列表
        data = []

        # 遍历每个文档，组装插入数据（i--》文档索引，doc--》当前文档对象）
        for i, doc in enumerate(documents):
            # 生成文档的唯一ID，使用哈希值
            # encode('utf-8')将字符串编码为UTF-8字节序列-->md5需要字节
            # hexdigest():将哈希结果转成16进制字符串（32位，适合作为ID）
            text_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
            # 处理稀疏向量：BME-M3返回的稀疏向量是矩阵，需要转换成Milvus支持的字典格式
            sparse_vector = {}  # 初始化稀疏字典
            # 获取第i个文档的稀疏向量行，getrow(i)--》获取矩阵的第i行
            row = embeddings['sparse'][[i]]
            # print(row)

            # 获取稀疏向量的非零值索引和对应的权重
            indices = row.indices
            # print(indices)
            values = row.data

            # 组装稀疏向量字典，将索引和权重配对
            for idx, value in zip(indices, values):
                sparse_vector[idx] = value
                # print(f'稀疏向量字典: {sparse_vector}')

                # 查看稠密向量及其维度
                # print(f'稠密向量: {embeddings["dense"][i]}, 维度: {len(embeddings["dense"][i])}')

            # 组装单条插入数据--》字段要和集合Schema一致
            data.append({
                'id': text_hash,  # 唯一ID（md5哈希值）
                'text': doc.page_content,  # 文档内容
                'dense_vector': embeddings['dense'][i],  # 稠密向量（BGE-M3生成）
                'sparse_vector': sparse_vector,  # 稀疏向量（组装后的字典格式）
                'parent_id': doc.metadata['parent_id'],  # 父块ID（从元数据中获取）
                'parent_content': doc.metadata['parent_content'],  # 父块内容
                'source': doc.metadata.get('source', 'unknown'),  # 学科类别
                'timestamp': doc.metadata.get('timestamp', 'unknown')  # 时间戳
            })

        # 插入数据到Milvus--》仅当data非空时
        if data:
            # 使用upsert操作:相同ID则更新，不存在则插入
            self.client.upsert(collection_name=self.collection_name, data=data)
            # 记录日志
            self.logger.info(f'已插入/更新 {len(data)} 条文档到集合{self.collection_name}')
        else:
            self.logger.warning('数据为空，未插入任何文档')

    # todo 3.4定义函数--》混合检索（稀疏+稠密）+ 重排序，返回精准的父文档
    def hybrid_retrieve_and_reorder(self, query, k=conf.RETRIEVAL_K, source_filter=None):
        """
        该函数用于执行：混合检索（稀疏+稠密）+ 重排序，返回精准的父文档
        :param query:用户查询文本
        :param k:混合检索返回的top-k子块数量
        :param source_filter:学科过滤
        :return:重排序后的top-m个父文档列表
        """
        # 使用BGE-M3嵌入函数生成查询向量
        query_embedding = self.embedder_function([query])
        # 获取查询的稠密向量
        query_dense_vector = query_embedding['dense'][0]
        # 初始化查询的稀疏向量字典
        query_sparse_vector = {}
        # 获取查询稀疏向量的第0行数据
        row = query_embedding['sparse'][[0]]
        # 获取稀疏向量的非零值索引
        indices = row.indices
        # 获取稀疏向量的非零值权重
        values = row.data
        # 组装稀疏向量字典，将索引和权重配对
        for idx, value in zip(indices, values):
            query_sparse_vector[idx] = value

        # 构建检索过滤表达式：按学科过滤--》若source-filter存在，
        if source_filter:
            filter_expr = f"source == '{source_filter}'"
        else:
            filter_expr = ""
        # print(f"检索过滤表达式: {filter_expr}")

        # 构建稠密向量检索请求：定义稠密向量的检索参数
        dense_request = AnnSearchRequest(
            data=[query_dense_vector],  # 查询向量，列表格式
            anns_field='dense_vector',  # 检索的向量字段
            param={'metric_type': 'IP', 'params': {'nprobe': 10}},
            limit=k,
            expr=filter_expr  # 应用过滤表达式
        )

        # 构建稀疏向量检索请求:定义稀疏向量的检索参数
        sparse_request = AnnSearchRequest(
            data=[query_sparse_vector],  # 查询向量，列表格式
            anns_field='sparse_vector',  # 检索的向量字段
            param={'metric_type': 'IP', 'params': {}},  # 稀疏向量无需额外参数，只需要指定内积相似度
            limit=k,
            expr=filter_expr  # 应用过滤表达式
        )

        # 创建加权排序器--》融合稀疏和稠密向量的检索结果，按权重计算最终得分
        # 权重逻辑：稠密向量侧重语义相似度，稀疏向量侧重关键词匹配
        ranker = WeightedRanker(1.0, 0.7)  # 稀疏向量权重1.0，稠密向量权重0.7

        # 执行混合检索--》获取稠密和稀疏向量的检索结果，返回top-k个子块
        result = self.client.hybrid_search(
            collection_name=self.collection_name,  # 目标集合
            reqs=[dense_request, sparse_request],  # 稠密和稀疏向量的检索请求列表
            ranker=ranker,  # 加权排序器
            limit=k,
            output_fields=['text', 'parent_id', 'parent_content', 'source', 'timestamp']
        )[0]  # 因为就一个查询，所以返回结果列表的第一个元素
        # print(f'result: {result}, type: {type(result)}，数据量: {len(result)}')

        # 将检索结果转成langchain document对象--》统一格式
        child_chunks = [self._doc_from_hit(hit['entity']) for hit in result]
        # print(f'子块列表: {child_chunks}，数据量: {len(child_chunks)}')

        # 从子块中提取去重的父文档--》避免同一父块的多个子块重复返回
        parent_doc = self._get_unique_parent_documents(child_chunks)
        # print(f'父文档: {parent_doc}，数据量: {len(parent_doc)}')

        # 重排序逻辑：父文档数量《2跳过排序（无需优化）--》直接返回
        if len(parent_doc) < 2:
            return parent_doc[:conf.CANDIDATE_M]

        # 父文档数量》=2时，执行重排序，提升相关性
        if parent_doc:
            # 构建“查询-文档”配对列表，重排序模型需要改格式输入，每个配对未[query, doc_content]
            pairs = [[query, doc.page_content] for doc in parent_doc]
            # 计算相关得分
            scores = self.reranker.predict(pairs)
            # print(f'相关得分: {scores}')

            # 按得分降序排序--》将得分和父文档配对，按得分从高到低排序
            ranked_parent_docs = [doc for _,doc in sorted(zip(scores, parent_doc),key=lambda x:x[0] ,reverse=True)]
        else:
            # 若父文档列表为空，返回空列表
            ranked_parent_docs = []
        # 返回重排序后的父文档列表--》top-m个
        return ranked_parent_docs[:conf.CANDIDATE_M]

    # todo 3.5定义函数--》从子块列表中提取去重的父文档
    def _get_unique_parent_documents(self, child_chunks):
        # 初始化集合，用于存储去重的父文档
        unique_content = set()
        # 初始化列表，用于存储唯一父文档
        unique_docs = []
        # 遍历所有子块
        for chunk in child_chunks:
            # 获取子块中的父块内容，默认为子块内容
            parent_content = chunk.metadata.get('parent_content', chunk.page_content)
            # 检查父块内容是否非空且未重复
            if parent_content and parent_content not in unique_content:
                # 创建新的Document对象，包含父块内容和元数据
                unique_docs.append(Document(page_content=parent_content, metadata=chunk.metadata))
                # 将父文档添加到去重集合中
                unique_content.add(parent_content)
        # 返回去重后的父文档列表
        return unique_docs

    # todo 3.6定义函数--》将Milvus结果（hit）转换为LangChain Document对象
    def _doc_from_hit(self, hit):
        # 创建并返回LangChain Document对象，填充内容和元数据
        return Document(
            page_content=hit['text'],
            metadata={
                'parent_id': hit['parent_id'],
                'parent_content': hit['parent_content'],
                'source': hit['source'],
                'timestamp': hit['timestamp']
            }
        )


# todo 4.测试代码
if __name__ == '__main__':
    # 实例化VectorStore对象--》自动连接Milvus，创建/加载集合
    vector_store = VectorStore()

    # 向量入库
    # # 定义变量，记录文档路径
    # directory_path = '../data/ai_data'
    # # {'dense': 1024, 'colbert_vecs': 1024, 'sparse': 250002}
    # # dense:稠密向量--》抓整体语义，避免因’关键词不同但意思相近‘漏掉相关文档
    # # colbert_vecs--》抓词级上下文，解决'同词不同义'问题
    # # sparse稀疏--》抓关键词权重，确保特定术语的准确性和相关性
    # # print(vector_store.embedder_function.dim)
    #
    # # 获取所有处理后的文档
    # documents = process_documents(directory_path)
    # # 添加文档到Milvus中
    # vector_store.add_documents(documents)

    # 混合检索+结果重排
    query = "如何使用Python进行数据处理？"
    results = vector_store.hybrid_retrieve_and_reorder(query, k=5, source_filter='ai')
    # 打印检索结果数量和具体文档内容
    print(f"检索结果数量: {len(results)}")
    print(f"具体文档内容: {results}")
