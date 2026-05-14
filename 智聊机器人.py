from dotenv import load_dotenv
load_dotenv()
import os
import streamlit as st
from PIL import Image
import io
import base64
import requests

from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
import time

# ====================== 页面配置 ======================
st.set_page_config(
    page_title="智聊机器人",
    page_icon="🤖",
    layout="wide"
)
st.title("🤖 智聊机器人 ")

# ====================== 初始化模型 & 工具 ======================
model = init_chat_model(
    model="qwen3.5-plus",
    model_provider="openai",
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    api_key=os.getenv("DASHSCOPE_API_KEY")
)

web_search = TavilySearch(max_results=5, topic="general")
tools = [web_search]

system_prompt = """
你是一个智能聊天机器人，喜欢跟用户聊天，
当用户有问题问你的时候，你擅于思考并调用web_search工具进行查询相关信息给予回答，
当网上查不到相关信息时你不会盲目回答，而是自己推理答案回答，并且告诉用户你查找不到相关资料，只能提供一个仅供参考的答案
若用户给你提供照片，首先识别照片中的核心问题所在，并结合用户需求进行答案搜索
"""

# 移除所有checkpointer，直接创建agent
agent = create_react_agent(
    model=model,
    tools=tools,
    prompt=system_prompt
)

# ====================== 图片识别函数 ======================
def recognize_image(image):
    try:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {os.getenv('DASHSCOPE_API_KEY')}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "qwen-vl-plus",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "请简洁描述这张图片的内容"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                    ]
                }
            ]
        }
        # 延长超时+增加重试
        for attempt in range(2):
            try:
                resp = requests.post(url, json=data, headers=headers, timeout=60)
                resp.raise_for_status()
                break
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
                if attempt == 1:
                    raise
                time.sleep(2)
        result = resp.json()

        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        else:
            return "图片识别失败，接口返回异常"
    except Exception as e:
        return ""

# ====================== 聊天历史（核心：用session_state存储完整对话）======================
if "messages" not in st.session_state:
    st.session_state.messages = []

# 渲染历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("image"):
            st.image(msg["image"], caption="上传的图片", width=300)

# ====================== 上传图片 ======================
uploaded_file = st.file_uploader("上传图片（可选）", type=["png", "jpg", "jpeg"])
image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="已上传", width=300)

# ====================== 输入框 ======================
user_input = st.chat_input("输入你的问题...")

if user_input or uploaded_file:
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(user_input if user_input else "用户上传了图片")
        if image:
            st.image(image, width=300)

    # 处理图片
    image_desc = ""
    if image:
        with st.spinner("正在识别图片..."):
            image_desc = recognize_image(image)
            if image_desc:
                st.info(f"图片识别结果：{image_desc}")
            else:
                st.warning("图片识别超时，将忽略图片内容继续对话")

    # 构造当前用户消息
    current_user_msg = ""
    if user_input:
        current_user_msg += user_input
    if image_desc:
        current_user_msg += f"\n【图片内容】：{image_desc}"

    if not current_user_msg.strip():
        st.error("请输入问题或上传图片后再发送！")
        st.stop()

    # 构造agent需要的完整历史消息（核心：把所有历史+当前消息都传给agent）
    agent_messages = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            agent_messages.append(("user", msg["content"]))
        elif msg["role"] == "assistant":
            agent_messages.append(("assistant", msg["content"]))
    # 加入当前用户的新消息
    agent_messages.append(("user", current_user_msg))

    # AI 回答
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                # 传入完整历史消息，无需任何checkpointer和config
                response = agent.invoke({"messages": agent_messages})
                if "messages" in response and len(response["messages"]) > 0:
                    answer = response["messages"][-1].content
                else:
                    answer = "抱歉，我无法回答这个问题，请重试。"
            except Exception as e:
                answer = f"出错了：{str(e)}"
                st.error(answer)

            st.markdown(answer)

    # 保存到session_state（Streamlit会话内永久保留，刷新页面不丢失）
    st.session_state.messages.append({
        "role": "user",
        "content": current_user_msg,
        "image": image if image else None
    })
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
