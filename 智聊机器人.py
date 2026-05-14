from dotenv import load_dotenv

load_dotenv()
import os
import streamlit as st
from PIL import Image
import io
import base64
import requests
import sqlite3

from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent

# ====================== 页面配置 ======================
st.set_page_config(
    page_title="智聊机器人",
    page_icon="🤖",
    layout="wide"
)
st.title("🤖 智聊机器人 - 支持图片上传")

# ====================== 初始化模型 & 工具 ======================
model = init_chat_model(
    model="qwen3.5-plus",
    model_provider="openai",
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    api_key=os.getenv("DASHSCOPE_API_KEY")
)

web_search = TavilySearch(max_results=5, topic="general")
tools = [web_search]

# 数据库记忆
connection = sqlite3.connect("resources/personal_chief.db", check_same_thread=False)
checkpointer = SqliteSaver(connection)

# 智能体（改用稳定的 create_react_agent）
system_prompt = """
你是一个智能聊天机器人，喜欢跟用户聊天，
当用户有问题问你的时候，你擅于思考并调用web_search工具进行查询相关信息给予回答，
当网上查不到相关信息时你不会盲目回答，而是自己推理答案回答，并且告诉用户你查找不到相关资料，只能提供一个仅供参考的答案
若用户给你提供照片，首先识别照片中的核心问题所在，并结合用户需求进行答案搜索
"""
agent = create_react_agent(
    model=model,
    tools=tools,
    checkpointer=checkpointer,
    prompt=system_prompt
)

config = {"configurable": {"thread_id": "streamlit_chat"}}


# ====================== 图片识别函数 ======================
def recognize_image(image):
    try:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # 改用兼容OpenAI的接口，更稳定
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
        # 增加超时和错误捕获
        resp = requests.post(url, json=data, headers=headers, timeout=20)
        resp.raise_for_status()
        result = resp.json()

        # 安全地获取返回结果
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        else:
            return f"图片识别失败，接口返回异常：{result}"
    except requests.exceptions.RequestException as e:
        return f"网络请求失败：{str(e)}"
    except Exception as e:
        return f"图片识别出错：{str(e)}"
# ====================== 聊天历史 ======================
if "messages" not in st.session_state:
    st.session_state.messages = []

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

    # 构造提问
    prompt = ""
    if user_input:
        prompt += user_input
    if image_desc:
        prompt += f"\n【图片内容】：{image_desc}"

    if not prompt.strip():
        st.error("请输入问题或上传图片后再发送！")
        st.stop()

    # AI 回答
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                response = agent.invoke(
                    {"messages": [("user", prompt)]},
                    config=config
                )
                # 提取最终回答
                if "messages" in response and len(response["messages"]) > 0:
                    answer = response["messages"][-1].content
                else:
                    answer = "抱歉，我无法回答这个问题，请重试。"
            except Exception as e:
                answer = f"出错了：{str(e)}"
                st.error(answer)

            st.markdown(answer)

    # 保存历史
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "image": image if image else None
    })
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })