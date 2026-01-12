import { useState, useRef, useEffect, useCallback } from 'react';
import { ChatMessage, AppStatus, ServerMessage } from '@/types';
import { translations } from '@/locales';

interface UseAudioChatProps {
  t: typeof translations['en'];
}

export const useAudioChat = ({ t }: UseAudioChatProps) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [status, setStatus] = useState<AppStatus>('idle');
  const [isConnected, setIsConnected] = useState(false);

  const websocketRef = useRef<WebSocket | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioQueueRef = useRef<string[]>([]);
  const isPlayingRef = useRef(false);
  const sendQueueRef = useRef<Promise<void>>(Promise.resolve());

  // Blob to Base64 helper
  const blobToBase64 = (blob: Blob): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => {
        if (typeof reader.result === 'string') {
          resolve(reader.result.split(',')[1]);
        } else {
          reject(new Error('Failed to convert blob to base64'));
        }
      };
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  };

  // 初始化 WebSocket
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws/chat');

    ws.onopen = () => {
      console.log('🔗 Connected to Auralis Backend');
      setIsConnected(true);
    };

    ws.onclose = () => {
      console.log('🔌 Disconnected');
      setIsConnected(false);
    };

    ws.onmessage = async (event) => {
      try {
        const data: ServerMessage = JSON.parse(event.data);
        handleServerMessage(data);
      } catch (e) {
        console.error('Failed to parse WebSocket message:', e);
      }
    };

    websocketRef.current = ws;

    return () => {
      ws.close();
    };
  }, []);

  // 处理服务器消息
  const handleServerMessage = (data: ServerMessage) => {
    switch (data.type) {
      case 'user-message':
        // 更新用户消息（替换占位符或添加新消息）
        setMessages((prev) => {
          const lastMsg = prev[prev.length - 1];
          if (lastMsg && lastMsg.role === 'user' && lastMsg.text === t.welcome.user_placeholder) {
            // 替换占位符
            return [
              ...prev.slice(0, -1),
              { ...lastMsg, text: data.content }
            ];
          }
          // 添加新的用户消息
          return [...prev, { role: 'user', text: data.content }];
        });
        break;

      case 'text-update':
        // AI 文本流式更新
        setMessages((prev) => {
          const lastMsg = prev[prev.length - 1];
          if (lastMsg && lastMsg.role === 'ai') {
            return [
              ...prev.slice(0, -1),
              { ...lastMsg, text: lastMsg.text + data.content }
            ];
          }
          return [...prev, { role: 'ai', text: data.content }];
        });
        break;

      case 'audio-chunk':
        audioQueueRef.current.push(data.content);
        playAudioQueue();
        break;

      case 'status':
        if (data.content === 'idle' && audioQueueRef.current.length > 0) {
            // 队列未播完，暂不设置 idle，由播放器控制
        } else {
            setStatus(data.content as AppStatus);
        }
        break;
    }
  };

  // 播放队列
  const playAudioQueue = async () => {
    if (isPlayingRef.current || audioQueueRef.current.length === 0) return;

    isPlayingRef.current = true;
    setStatus('playing');

    const chunk = audioQueueRef.current.shift();
    if (!chunk) return;

    const audio = new Audio(`data:audio/wav;base64,${chunk}`);
    
    audio.onended = () => {
      isPlayingRef.current = false;
      if (audioQueueRef.current.length > 0) {
        playAudioQueue();
      } else {
        setStatus('idle');
      }
    };

    try {
      await audio.play();
    } catch (err) {
      console.error('Audio playback failed:', err);
      isPlayingRef.current = false;
      setStatus('idle');
    }
  };

  // 开始录音
  const startRecording = useCallback(async () => {
    if (!websocketRef.current || websocketRef.current.readyState !== WebSocket.OPEN) {
      alert(t.errors.backend_offline);
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      mediaRecorderRef.current = mediaRecorder;

      // Reset send queue for new recording session
      sendQueueRef.current = Promise.resolve();

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          const blob = event.data;
          sendQueueRef.current = sendQueueRef.current.then(async () => {
            if (websocketRef.current?.readyState === WebSocket.OPEN) {
              try {
                const base64 = await blobToBase64(blob);
                websocketRef.current.send(JSON.stringify({ 
                  type: 'audio-chunk', 
                  content: base64 
                }));
              } catch (error) {
                console.error('Error processing audio chunk:', error);
              }
            }
          });
        }
      };

      mediaRecorder.onstop = () => {
        sendQueueRef.current = sendQueueRef.current.then(() => {
          if (websocketRef.current?.readyState === WebSocket.OPEN) {
            websocketRef.current.send(JSON.stringify({ type: 'audio-end' }));
          }
        });
      };

      mediaRecorder.start(200);
      setStatus('recording');
      
      setMessages(prev => [...prev, { role: 'user', text: t.welcome.user_placeholder }]);

    } catch (err) {
      console.error('Microphone access denied:', err);
      alert(t.errors.mic_access);
    }
  }, [t]);

  // 停止录音
  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
      
      setStatus('processing');
    }
  }, []);

  // 发送文本消息
  const sendTextMessage = useCallback((text: string) => {
    if (!websocketRef.current || websocketRef.current.readyState !== WebSocket.OPEN) {
      alert(t.errors.backend_offline);
      return;
    }

    const trimmedText = text.trim();
    if (!trimmedText) {
      return;
    }

    // 发送到后端（后端会发送 user-message 回来，所以这里不直接添加）
    websocketRef.current.send(JSON.stringify({ 
      type: 'text-input', 
      content: trimmedText 
    }));
  }, [t]);

  return {
    messages,
    status,
    isConnected,
    startRecording,
    stopRecording,
    sendTextMessage
  };
};