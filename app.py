# todo 1.导包

# 导入FastAPI相关模块，用于构建API和WebSocket-->构建http接口，WebSocket长连接
from fastapi import FastAPI, WebSocket, HTTPException, Query, Depends
# 导入FastAPI响应类型，用于流式响应和文件服务
from fastapi.responses import StreamingResponse, FileResponse
# 导入CORS中间件，支持跨域请求
from fastapi.middleware.cors import CORSMiddleware
# 导入静态文件服务模块
from fastapi.staticfiles import StaticFiles
# 导入WebSocket断开隔离
from fastapi.websockets import WebSocketDisconnect
import os
# 导入Pydantic相关模块，用于数据验证和序列化
from pydantic import BaseModel
# 导入异步事件循环模块
import asyncio
# 导入JSON模块，用于数据序列化和反序列化
import json
# 导入UUID模块，用于生成唯一标识符
import uuid
# 导入类型注解模块
from typing import List, Dict, Any, Optional
# 导入时间模块
import time
# 导入正则表达式模块，用于字符串匹配和处理
import re
# 导入优化后的问答系统
from new_main import IntegratedQASystem

# todo 2.FastAPI应用实例初始化和全局中间件配置
# 创建FastAPI的应用实例，设置接口文档标题，接口整体业务描述，自动生成Swagger在线文档
app = FastAPI(title='问答系统API', description='这是一个优化后的问答系统API')

# 全局注入CORS跨域中间件，配置前后端跨域请求规则
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源的跨域请求
    allow_credentials=True,  # 允许发送Cookie
    allow_methods=["*"],  # 允许所有HTTP方法--》放行全部http请求，如：GET, POST, PUT, DELETE等
    allow_headers=["*"],  # 允许所有请求头
)

# 创建静态资源存储目录static， exist_ok=True表示如果目录已存在则不会报错
os.makedirs("static", exist_ok=True)

# 全局实例化优化后的问答系统
qa_system = IntegratedQASystem()

# todo 3.全局固定问候语，正则匹配规则配置
GREETING_PATTERNS = [
    {
        "pattern": r"^(你好|您好|hi|hello)",  # 匹配问候语
        "response": "你好！我是黑马程序员，专注于为学生答疑解惑，很高兴为你服务！"
    },
    {
        "pattern": r"^(你是谁|您是谁|你叫什么|你的名字|who are you)",  # 匹配身份询问
        "response": "我是黑马程序员，你的智能学习助手，致力于提供 IT 教育相关的解答！"
    },
    {
        "pattern": r"^(在吗|在不在|有人吗)",  # 匹配在线确认
        "response": "我在！我是黑马程序员，随时为你解答问题！"
    },
    {
        "pattern": r"^(干嘛呢|你在干嘛|做什么)",  # 匹配状态询问
        "response": "我正在待命，随时为你解答 IT 学习相关的问题！有什么我可以帮你的？"
    }
]


# todo 4.Pydantic请求/响应数据模型定义
# todo 4.1非流式查询接口入参结构体--》接收post请求json参数
# 定义查询请求模型
class QueryRequest(BaseModel):
    query: str  # 查询内容，必填
    source_filter: Optional[str] = None  # 学科过滤，可选
    session_id: Optional[str] = None  # 会话 ID，可选


# todo 4.2 非流式查询接口标准返回结构体 -> 统一接口出参格式.
# 定义查询响应模型
class QueryResponse(BaseModel):
    answer: str  # 答案内容 -> 问答最终回复内容.
    is_streaming: bool  # 是否流式响应 -> 标签当前回复是否需要流式输出, True代表需切换WebSocket接口
    session_id: str  # 会话 ID -> 用于上下文历史关联
    processing_time: float  # 处理时间 -> 本次问答完整处理耗时, 单位: 秒.


# todo 5.静态资源挂载与页面路由配置
# 将本地static目录挂载至服务器根目录
app.mount("/static", StaticFiles(directory="static"), name="static")


# 根据根路径GET接口，访问首页HTML页面
@app.get("/")  # 这个/表示：http://ip地址:端口/
async def read_root():
    # 返回static目录下的首页html文件--》实现域名根路径直接打开前端界面
    return FileResponse("static/index.html")


# todo 5.HTTP RestFul业务接口定义
# 创建新会话接口
# TODO 类似于java中的@PostMapping
@app.post("/api/create_session")
async def create_session():
    session_id = str(uuid.uuid4())  # 生成唯一会话 ID
    return {"session_id": session_id}  # 返回会话 ID


# todo  5.2 查询历史消息接口
# TODO restful api 规范 。比如某厂规范：internal-api.xxxx.com/{业务线: driver/passanger}/{二级:运营、打车}/xxxx
# TODO 对外的api(openapi) :  /api/v1/edu-rag/history/{session_id}
@app.get("/api/history/{session_id}")
async def get_history(session_id: str):
    try:
        # 获取指定会话的历史记录
        history = qa_system.get_conversation_history(session_id)
        # 返回会话 ID 和历史记录
        # TODO 规范操作：{"data": {"session_id": session_id, "history": history} ,"errno":0,"errmsg":报错信息, "log_id":日志编号}
        return {"session_id": session_id, "history": history}
    except Exception as e:
        # 抛出 HTTP 异常，包含错误信息
        raise HTTPException(status_code=500, detail=f"获取历史记录失败: {str(e)}")


# @ todo 5.3 清除历史消息接口 -> 根据会话ID清空历史对话接口.
@app.delete("/api/history/{session_id}")
async def clear_history(session_id: str):
    # 清除指定会话的历史记录
    success = qa_system.clear_session_history(session_id)
    if success:
        # 返回成功状态
        return {"status": "success", "message": "历史记录已清除"}
    else:
        # 抛出 HTTP 异常
        raise HTTPException(status_code=500, detail="清除历史记录失败")


# todo 6. 公共工具函数 -> 检测用户输入是否为问候短句, 匹配则返回预设回顾 -: 检查是否为日常问候用语并返回模板回复
def check_greeting(query: str) -> Optional[str]:
    query_text = query.strip()  # 去除首尾空格
    for pattern_info in GREETING_PATTERNS:
        # 使用正则匹配，忽略大小写, 兼容中英文大小写混用场景.
        if re.match(pattern_info["pattern"], query_text, re.IGNORECASE):
            return pattern_info["response"]  # 返回匹配的回复
    return None  # 无匹配返回 None


# todo 7. 同步非主流式问答POST接口, 一次性返回完整回复答案, 不支持分段流式输出.
# 非流式查询接口
@app.post("/api/query")
async def query(request: QueryRequest):
    start_time = time.time()  # 记录开始时间
    # 使用请求中的 session_id 或生成新 ID
    session_id = request.session_id or str(uuid.uuid4())
    # 检查是否为日常问候
    greeting_response = check_greeting(request.query)
    if greeting_response:
        # 返回问候回复
        return {
            "answer": greeting_response,
            "is_streaming": False,
            "session_id": session_id,
            "processing_time": time.time() - start_time
        }
    # 执行 BM25 搜索
    # 注意：search() 的第二个返回值是"是否命中"（命中：(答案, True)；未命中：(None, False)），
    # 不是"是否需要 RAG"，两者含义相反；写反会把 None 当答案返回给前端，导致 marked.parse(null) 报错。
    answer, hit = qa_system.bm25_search.search(request.query, threshold=0.85)
    if not hit:
        # 未命中 → 需要走 RAG，让前端改用 WebSocket 流式接口
        return {
            "answer": "",
            "is_streaming": True,
            "session_id": session_id,
            "processing_time": time.time() - start_time
        }
    # 命中 → 直接返回 MySQL 或 Redis 缓存里的答案
    return {
        "answer": answer,
        "is_streaming": False,
        "session_id": session_id,
        "processing_time": time.time() - start_time
    }


# 流式查询 WebSocket 接口
# WebSocket长连接流式问答接口, 逐Token分段推送答案, 实现: 打字机流式效果.
# todo 8. 这里用的还是fastapi框架，但是这里接受的对象不会转成我们定义的结构体。 使用的是WebSocket
@app.websocket("/api/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()  # 接受 WebSocket 连接
    try:
        while True:
            # 接收客户端消息
            data = await websocket.receive_text()
            request_data = json.loads(data)  # 解析 JSON 数据
            # 获取查询参数
            query = request_data.get("query")
            source_filter = request_data.get("source_filter")
            session_id = request_data.get("session_id", str(uuid.uuid4()))
            start_time = time.time()  # 记录开始时间
            # 发送开始标志
            if websocket.client_state == websocket.client_state.CONNECTED:
                await websocket.send_json({
                    "type": "start",
                    "session_id": session_id
                })
            # 检查是否为日常问候
            greeting_response = check_greeting(query)
            if greeting_response:
                if websocket.client_state == websocket.client_state.CONNECTED:
                    # 发送问候回复
                    await websocket.send_json({
                        "type": "token",
                        "token": greeting_response,
                        "session_id": session_id
                    })
                    # 发送结束标志
                    await websocket.send_json({
                        "type": "end",
                        "session_id": session_id,
                        "is_complete": True,
                        "processing_time": time.time() - start_time
                    })
                break
            # 调用问答系统，流式处理查询
            collected_answer = ""
            for token, is_complete in qa_system.query(query, source_filter=source_filter, session_id=session_id):
                collected_answer += token  # 累积答案
                if is_complete and not collected_answer:
                    if websocket.client_state == websocket.client_state.CONNECTED:
                        # 发送结束标志
                        await websocket.send_json({
                            "type": "end",
                            "session_id": session_id,
                            "is_complete": True,
                            "processing_time": time.time() - start_time
                        })
                    break
                if token and websocket.client_state == websocket.client_state.CONNECTED:
                    # 发送 token 数据
                    await websocket.send_json({
                        "type": "token",
                        "token": token,
                        "session_id": session_id
                    })
                if is_complete:
                    if websocket.client_state == websocket.client_state.CONNECTED:
                        # 发送结束标志
                        await websocket.send_json({
                            "type": "end",
                            "session_id": session_id,
                            "is_complete": True,
                            "processing_time": time.time() - start_time
                        })
                    break
                await asyncio.sleep(0.01)  # 控制流式输出的速度
    except WebSocketDisconnect as e:
        # 记录 WebSocket 断开信息
        print(f"WebSocket disconnected: code={e.code}, reason={e.reason}")
    except Exception as e:
        # 记录错误信息
        print(f"WebSocket error: {str(e)}")
        # 客户端可能已经断开，此时 send_json 会再抛一次，必须包住
        try:
            if websocket.client_state == websocket.client_state.CONNECTED:
                # 发送错误消息
                await websocket.send_json({
                    "type": "error",
                    "error": str(e)
                })
        except Exception:
            pass
    finally:
        try:
            if websocket.client_state == websocket.client_state.CONNECTED:
                # 关闭 WebSocket 连接
                await websocket.close()
        except Exception as e:
            # 记录关闭连接时的错误
            print(f"Error closing WebSocket: {str(e)}")

# todo 9. 健康检查接口 -> 运维监控接口(了解即可)
# TODO k8s: 管理和部署容器， 基于docker(其他虚拟化框架)一个管理工具
# TODO 探针机制(去调用接口知道当前的服务它的状态)： 就绪探针(探测是不是启动)、 存活探针(是不是正常服务)
@app.get("/health")
async def health_check():
    # 返回健康状态标记, k8s调用该接口返回200则判断服务正常运行.
    return {"status": "healthy"}  # 返回健康状态

# 获取有效学科类别接口
@app.get("/api/sources")
async def get_sources():
    return {"sources": qa_system.config.VALID_SOURCES}  # 返回学科类别列表


# 主程序入口
if __name__ == "__main__":
    # springboot = springcore + spRirngmvc + tomcat
    # fastapi = springmvc (url -> 方法调用)
    # uvicorn = tomcat (服务容器，负责处理多线程、高并发等)

    import uvicorn      # uvicorn库 -> 异步web服务 容器, 用于启动FastAPI应用.
    import os

    # 从环境变量获取主机和端口，默认值为 0.0.0.0:8080
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 8080))

    # 运行 FastAPI 应用，监听指定的主机和端口
    # reload=False 关闭热重载, 生产环境禁用, 避免性能损耗.
    uvicorn.run("app:app", host=host, port=port, reload=False)
