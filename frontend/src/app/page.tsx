"use client";

import React, { useRef, useEffect, useState } from "react";
import { Mic, Square, Loader2, Sparkles, Languages, Radio, MessageSquare, Send } from "lucide-react";
import { useAudioChat } from "@/hooks/useAudioChat";
import { useLanguage } from "@/components/LanguageProvider";
import { cn } from "@/lib/utils";

export default function Home() {
  const { t, locale, setLocale } = useLanguage();
  const { messages, status, isConnected, startRecording, stopRecording, sendTextMessage } = useAudioChat({ t });
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [inputMode, setInputMode] = useState<'voice' | 'text'>('voice');
  const [textInput, setTextInput] = useState('');
  const textInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, status]);

  useEffect(() => {
    if (inputMode === 'text' && textInputRef.current) {
      textInputRef.current.focus();
    }
  }, [inputMode]);

  const toggleLanguage = () => setLocale(locale === 'en' ? 'zh' : 'en');

  const handleTextSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (textInput.trim() && isConnected && status !== 'processing') {
      sendTextMessage(textInput);
      setTextInput('');
    }
  };

  // 状态颜色映射
  const getOrbColor = () => {
    switch (status) {
      case 'recording': return 'from-rose-500 to-orange-500 shadow-rose-500/50';
      case 'processing': return 'from-violet-500 to-fuchsia-500 shadow-violet-500/50';
      case 'playing': return 'from-emerald-400 to-cyan-500 shadow-emerald-500/50';
      default: return 'from-blue-500 to-indigo-500 shadow-blue-500/50'; // idle
    }
  };

  return (
    <main className="relative flex min-h-screen flex-col overflow-hidden bg-slate-950 font-sans text-slate-100">
      
      {/* 动态背景 */}
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        {/* 左上角光斑 */}
        <div className="absolute -left-[10%] -top-[10%] h-[500px] w-[500px] rounded-full bg-purple-500/20 blur-[100px] animate-blob" />
        {/* 右下角光斑 */}
        <div className="absolute -right-[10%] -bottom-[10%] h-[500px] w-[500px] rounded-full bg-blue-500/20 blur-[100px] animate-blob animation-delay-2000" />
        {/* 中间动态光斑 */}
        <div className={cn(
          "absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 h-[600px] w-[600px] rounded-full blur-[120px] transition-colors duration-1000 opacity-30",
          status === 'recording' ? "bg-red-500/30" : "bg-blue-500/10"
        )} />
      </div>

      {/* 顶部导航 */}
      <header className="fixed top-0 z-50 w-full border-b border-white/5 bg-slate-950/30 px-6 py-4 backdrop-blur-xl">
        <div className="mx-auto flex max-w-4xl items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-gradient-to-tr from-blue-500 to-purple-500 shadow-lg shadow-purple-500/20">
              <Sparkles size={16} className="text-white" />
            </div>
            <span className="text-lg font-semibold tracking-tight text-white/90">{t.title}</span>
          </div>

          <div className="flex items-center gap-4">
             <button 
                onClick={toggleLanguage}
                className="rounded-full p-2 text-slate-400 transition-colors hover:bg-white/10 hover:text-white"
              >
                <Languages size={18} />
              </button>
             
             {/* 极简状态点 */}
             <div className={cn("h-2 w-2 rounded-full transition-all duration-500", isConnected ? "bg-emerald-400 shadow-[0_0_10px_#34d399]" : "bg-rose-500")} />
          </div>
        </div>
      </header>

      {/* 聊天区域 */}
      <div className="relative z-10 flex-1 overflow-y-auto pt-28 pb-48 scrollbar-hide">
        <div className="mx-auto flex max-w-2xl flex-col gap-6 px-4">
          
          {messages.length === 0 && (
            <div className="mt-20 flex flex-col items-center justify-center space-y-6 text-center opacity-0 animate-in fade-in slide-in-from-bottom-5 duration-700 fill-mode-forwards">
              <div className="relative flex h-24 w-24 items-center justify-center rounded-3xl bg-gradient-to-br from-white/5 to-white/0 ring-1 ring-white/10 backdrop-blur-md">
                <Radio size={40} className="text-white/40" />
              </div>
              <p className="max-w-xs text-sm font-medium text-slate-400 leading-relaxed">
                {t.welcome.icon_text}
              </p>
            </div>
          )}

          {messages.map((msg, idx) => (
            <div
              key={idx}
              className={cn(
                "flex w-full animate-in fade-in slide-in-from-bottom-2 duration-300",
                msg.role === "user" ? "justify-end" : "justify-start"
              )}
            >
              <div
                className={cn(
                  "relative max-w-[85%] rounded-3xl px-6 py-4 text-[15px] leading-relaxed shadow-sm",
                  msg.role === "user"
                    ? "bg-gradient-to-br from-blue-600 to-indigo-600 text-white rounded-tr-sm"
                    : "bg-white/5 border border-white/5 text-slate-200 backdrop-blur-md rounded-tl-sm"
                )}
              >
                {msg.text}
              </div>
            </div>
          ))}

          {status === 'processing' && (
             <div className="flex w-full justify-start animate-pulse">
                <div className="flex items-center gap-3 rounded-3xl rounded-tl-sm border border-white/5 bg-white/5 px-6 py-4 backdrop-blur-md">
                   <Loader2 className="h-4 w-4 animate-spin text-purple-400" />
                   <span className="text-xs font-medium text-purple-200/70">{t.status.processing}</span>
                </div>
             </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <div className="fixed bottom-0 z-20 w-full bg-gradient-to-t from-slate-950 via-slate-950/80 to-transparent pt-12 pb-12">
        <div className="flex flex-col items-center justify-center gap-6">
          
          {/* Mode Toggle */}
          <div className="flex items-center gap-2 rounded-full bg-white/5 border border-white/10 p-1 backdrop-blur-md">
            <button
              onClick={() => setInputMode('voice')}
              className={cn(
                "flex items-center gap-2 rounded-full px-4 py-2 text-sm font-medium transition-all",
                inputMode === 'voice'
                  ? "bg-gradient-to-br from-blue-600 to-indigo-600 text-white shadow-lg"
                  : "text-slate-400 hover:text-white hover:bg-white/5"
              )}
            >
              <Mic size={16} />
              <span>{locale === 'en' ? 'Voice' : '语音'}</span>
            </button>
            <button
              onClick={() => setInputMode('text')}
              className={cn(
                "flex items-center gap-2 rounded-full px-4 py-2 text-sm font-medium transition-all",
                inputMode === 'text'
                  ? "bg-gradient-to-br from-blue-600 to-indigo-600 text-white shadow-lg"
                  : "text-slate-400 hover:text-white hover:bg-white/5"
              )}
            >
              <MessageSquare size={16} />
              <span>{locale === 'en' ? 'Text' : '文本'}</span>
            </button>
          </div>

          {/* Text Input Mode */}
          {inputMode === 'text' && (
            <form onSubmit={handleTextSubmit} className="w-full max-w-2xl px-4">
              <div className="flex items-center gap-3 rounded-3xl border border-white/10 bg-white/5 backdrop-blur-md px-4 py-3 focus-within:border-blue-500/50 focus-within:ring-2 focus-within:ring-blue-500/20 transition-all">
                <input
                  ref={textInputRef}
                  type="text"
                  value={textInput}
                  onChange={(e) => setTextInput(e.target.value)}
                  placeholder={locale === 'en' ? 'Type your message...' : '输入消息...'}
                  disabled={status === 'processing' || !isConnected}
                  className="flex-1 bg-transparent text-slate-200 placeholder:text-slate-500 outline-none disabled:opacity-50"
                />
                <button
                  type="submit"
                  disabled={!textInput.trim() || status === 'processing' || !isConnected}
                  className={cn(
                    "rounded-full p-2 transition-all",
                    textInput.trim() && isConnected && status !== 'processing'
                      ? "bg-gradient-to-br from-blue-600 to-indigo-600 text-white hover:scale-110 cursor-pointer"
                      : "bg-white/5 text-slate-500 cursor-not-allowed opacity-50"
                  )}
                >
                  <Send size={18} />
                </button>
              </div>
            </form>
          )}

          {/* Voice Input Mode */}
          {inputMode === 'voice' && (
            <>
              {/* 状态文本 */}
              <div className="h-6 text-sm font-medium tracking-widest uppercase transition-all duration-300">
                 {status === 'recording' && <span className="text-rose-400 drop-shadow-[0_0_8px_rgba(244,63,94,0.5)]">{t.status.recording}</span>}
                 {status === 'playing' && <span className="text-emerald-400 drop-shadow-[0_0_8px_rgba(52,211,153,0.5)]">{t.status.playing}</span>}
                 {status === 'idle' && <span className="text-slate-500">{t.status.idle}</span>}
              </div>

              {/* The Orb */}
              <button
                onMouseDown={startRecording}
                onMouseUp={stopRecording}
                onTouchStart={startRecording}
                onTouchEnd={stopRecording}
                onMouseLeave={status === 'recording' ? stopRecording : undefined}
                disabled={status === 'processing' || !isConnected}
                className={cn(
                  "group relative flex h-24 w-24 items-center justify-center rounded-full transition-all duration-500 ease-out",
                  isConnected ? "cursor-pointer hover:scale-105 active:scale-95" : "cursor-not-allowed opacity-50 grayscale"
                )}
              >
                {/* 外层光晕 (Idle时呼吸，Recording时剧烈) */}
                <div className={cn(
                   "absolute inset-0 rounded-full bg-gradient-to-br blur-xl transition-all duration-500",
                   getOrbColor(),
                   status === 'idle' && "animate-pulse opacity-40 group-hover:opacity-70",
                   status === 'recording' && "opacity-80 scale-150 duration-75", // 录音时变大
                   status === 'processing' && "animate-spin opacity-50 duration-[3s]"
                )} />

                {/* 实体球体 */}
                <div className={cn(
                   "relative flex h-full w-full items-center justify-center rounded-full bg-gradient-to-br shadow-inner ring-1 ring-white/20 backdrop-blur-sm transition-all duration-300",
                   getOrbColor()
                )}>
                  {status === 'recording' ? (
                     <Square fill="white" size={28} className="text-white drop-shadow-md" /> 
                  ) : (
                     <Mic fill="white" size={32} className={cn("text-white drop-shadow-md transition-transform duration-300", status==='processing' ? 'opacity-50' : 'group-hover:scale-110')} />
                  )}
                </div>

                {/* 波纹动画 (仅录音时) */}
                {status === 'recording' && (
                  <>
                     <div className="absolute inset-0 rounded-full border border-rose-500/50 animate-ping opacity-20 duration-1000" />
                     <div className="absolute inset-[-12px] rounded-full border border-rose-500/30 animate-ping opacity-10 animation-delay-300" />
                  </>
                )}
              </button>
            </>
          )}
        </div>
      </div>
      
      {/* 补充动画 */}
      <style jsx global>{`
        @keyframes blob {
          0% { transform: translate(0px, 0px) scale(1); }
          33% { transform: translate(30px, -50px) scale(1.1); }
          66% { transform: translate(-20px, 20px) scale(0.9); }
          100% { transform: translate(0px, 0px) scale(1); }
        }
        .animate-blob {
          animation: blob 7s infinite;
        }
        .animation-delay-2000 {
          animation-delay: 2s;
        }
      `}</style>
    </main>
  );
}