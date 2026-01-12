import os
import json
import base64
import asyncio
import uuid
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.core.asr import transcribe_audio
from app.core.llm import chat_stream
from app.core.tts import text_to_speech

router = APIRouter()

@router.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    """
    处理全双工语音对话的 WebSocket 端点
    """
    await websocket.accept()
    client_id = str(uuid.uuid4())[:8] # 给每个连接生成一个短ID方便日志查看
    print(f"🔌 Client connected: {client_id}")
    
    # 用于暂存接收到的音频切片
    audio_buffer = bytearray()
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "audio-chunk":
                chunk = base64.b64decode(message["content"])
                audio_buffer.extend(chunk)
            
            elif message["type"] == "text-input":
                # 处理文本输入
                user_text = message.get("content", "").strip()
                if not user_text:
                    continue
                
                print(f"👤 [{client_id}] User text: {user_text}")
                
                # 发送用户消息给前端
                await websocket.send_json({
                    "type": "user-message", 
                    "content": user_text
                })
                
                # 通知前端处理中
                await websocket.send_json({"type": "status", "content": "processing"})
                
                # 处理LLM响应
                sentence_buffer = ""
                punctuation = {",", "，", ".", "。", "?", "？", "!", "！", ";", "；", ":", "：", "\n"}
                
                try:
                    async for char in chat_stream(user_text):
                        # 实时推流文字
                        await websocket.send_json({"type": "text-update", "content": char})
                        
                        sentence_buffer += char
                        
                        # 断句
                        if char in punctuation:
                            if len(sentence_buffer.strip()) > 1:
                                print(f"🗣️ [{client_id}] Synthesizing: {sentence_buffer}")
                                audio_base64 = await text_to_speech(sentence_buffer)
                                
                                if audio_base64:
                                    await websocket.send_json({
                                        "type": "audio-chunk", 
                                        "content": audio_base64
                                    })
                                sentence_buffer = ""

                    # 处理剩余文本
                    if sentence_buffer.strip():
                         print(f"🗣️ [{client_id}] Synthesizing (Final): {sentence_buffer}")
                         audio_base64 = await text_to_speech(sentence_buffer)
                         if audio_base64:
                            await websocket.send_json({
                                "type": "audio-chunk", 
                                "content": audio_base64
                            })

                except Exception as e:
                    print(f"❌ LLM/TTS Process Error: {e}")
                    await websocket.send_json({"type": "text-update", "content": f"\n[Error: {str(e)}]"})
                
                await websocket.send_json({"type": "status", "content": "idle"})
            
            elif message["type"] == "audio-end":
                # 生成唯一文件名并保存
                request_id = str(uuid.uuid4())
                temp_audio_path = f"temp_input_{request_id}.webm"
                
                # 写入文件
                with open(temp_audio_path, "wb") as f:
                    f.write(audio_buffer)
                
                # 清空缓冲区
                audio_buffer = bytearray()
                
                # 通知前端
                await websocket.send_json({"type": "status", "content": "processing"})

                # ASR
                try:
                    # 使用 asyncio.to_thread 运行同步的 Whisper 识别
                    user_text = await asyncio.to_thread(transcribe_audio, temp_audio_path)
                    print(f"👂 [{client_id}] User said: {user_text}")
                except Exception as e:
                    print(f"❌ ASR Error: {e}")
                    user_text = ""

                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)

                # 如果没听到说话，直接跳过
                if not user_text.strip():
                    await websocket.send_json({"type": "status", "content": "idle"})
                    continue

                # 发送用户消息给前端（使用新的消息类型）
                await websocket.send_json({
                    "type": "user-message", 
                    "content": user_text
                })

                sentence_buffer = ""
                punctuation = {",", "，", ".", "。", "?", "？", "!", "！", ";", "；", ":", "：", "\n"}
                
                try:
                    async for char in chat_stream(user_text):
                        # 实时推流文字
                        await websocket.send_json({"type": "text-update", "content": char})
                        
                        sentence_buffer += char
                        
                        # 断句
                        if char in punctuation:
                            if len(sentence_buffer.strip()) > 1:
                                print(f"🗣️ [{client_id}] Synthesizing: {sentence_buffer}")
                                audio_base64 = await text_to_speech(sentence_buffer)
                                
                                if audio_base64:
                                    await websocket.send_json({
                                        "type": "audio-chunk", 
                                        "content": audio_base64
                                    })
                                sentence_buffer = ""

                    # 处理剩余文本
                    if sentence_buffer.strip():
                         print(f"🗣️ [{client_id}] Synthesizing (Final): {sentence_buffer}")
                         audio_base64 = await text_to_speech(sentence_buffer)
                         if audio_base64:
                            await websocket.send_json({
                                "type": "audio-chunk", 
                                "content": audio_base64
                            })

                except Exception as e:
                    print(f"❌ LLM/TTS Process Error: {e}")
                    await websocket.send_json({"type": "text-update", "content": f"\n[Error: {str(e)}]"})
                
                await websocket.send_json({"type": "status", "content": "idle"})

    except WebSocketDisconnect:
        print(f"👋 Client {client_id} disconnected")
    except Exception as e:
        print(f"❌ WebSocket Error: {e}")
        try:
            await websocket.close()
        except:
            pass