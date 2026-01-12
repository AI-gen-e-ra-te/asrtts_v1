export type Locale = 'en' | 'zh';

export const translations = {
  en: {
    title: 'Auralis',
    status: {
      connected: 'System Online',
      disconnected: 'Disconnected',
      recording: 'Recording Audio...',
      playing: 'Auralis Speaking...',
      idle: 'Hold to Speak',
      processing: 'Computing response...',
    },
    welcome: {
      icon_text: 'Ready to listen. Press the microphone to start.',
      user_placeholder: '...',
    },
    errors: {
      backend_offline: 'Backend disconnected. Please check the server.',
      mic_access: 'Microphone access denied.',
    },
  },
  zh: {
    title: 'Auralis',
    status: {
      connected: '系统在线',
      disconnected: '未连接',
      recording: '正在聆听...',
      playing: '正在回复...',
      idle: '按住说话',
      processing: '思考中...',
    },
    welcome: {
      icon_text: '随时待命。按下麦克风开始对话。',
      user_placeholder: '...',
    },
    errors: {
      backend_offline: '后端未连接，请检查服务是否启动',
      mic_access: '无法访问麦克风，请授权。',
    },
  },
};