export interface ChatMessage {
  role: 'user' | 'ai';
  text: string;
  isInterim?: boolean;
}

export type AppStatus = 'idle' | 'recording' | 'processing' | 'playing';

// WebSocket 发送的消息
export type ClientMessage = 
  | { type: 'audio-chunk'; content: string } // Base64 音频
  | { type: 'audio-end' }                     // 录音结束信号
  | { type: 'text-input'; content: string }; // 文本输入

// WebSocket 接收的消息
export type ServerMessage =
  | { type: 'text-update'; content: string } // AI 文本流式更新
  | { type: 'user-message'; content: string } // 用户消息（语音或文本输入）
  | { type: 'audio-chunk'; content: string } // TTS 音频片段
  | { type: 'status'; content: AppStatus };  // 状态变更