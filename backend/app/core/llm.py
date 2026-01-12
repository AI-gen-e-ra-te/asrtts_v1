import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

# 加载 .env 环境变量
load_dotenv()

# 从环境变量读取配置，提供默认值
BASE_URL = os.getenv("LLM_BASE_URL")
API_KEY = os.getenv("LLM_API_KEY")
MODEL = os.getenv("LLM_MODEL")

# 初始化异步客户端
client = AsyncOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

print(f"🧠 LLM Client initialized: {MODEL} @ {BASE_URL}")

async def chat_stream(prompt: str):
    """
    异步生成器：流式返回 LLM 的文本回复
    """
    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful voice assistant. Please keep your replies concise, short, and conversational suitable for TTS."},
                {"role": "user", "content": prompt}
            ],
            stream=True,
            temperature=0.7,
        )

        # 逐块读取流
        async for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    except Exception as e:
        print(f"❌ LLM Error: {e}")
        yield f" Error: {str(e)}"