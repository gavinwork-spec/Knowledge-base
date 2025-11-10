import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Send,
  Paperclip,
  Mic,
  MicOff,
  Download,
  RefreshCw,
  Copy,
  Check,
  Bot,
  User,
  Sparkles
} from 'lucide-react';
import { Button } from './button';
import { cn } from '@lib/utils';

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  isStreaming?: boolean;
  sources?: any[];
  quickActions?: string[];
}

interface ChatProps {
  messages: ChatMessage[];
  onSendMessage: (message: string) => void;
  isLoading?: boolean;
  placeholder?: string;
  showQuickActions?: boolean;
  manufacturingContext?: any;
  className?: string;
}

const Chat: React.FC<ChatProps> = ({
  messages,
  onSendMessage,
  isLoading = false,
  placeholder = "Ask about manufacturing procedures, safety guidelines, or technical specifications...",
  showQuickActions = true,
  manufacturingContext,
  className
}) => {
  const [input, setInput] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Handle input submission
  const handleSubmit = (e?: React.FormEvent) => {
    e?.preventDefault();
    if (input.trim() && !isLoading) {
      onSendMessage(input.trim());
      setInput('');
    }
  };

  // Handle voice recording (mock implementation)
  const handleVoiceRecording = () => {
    setIsRecording(!isRecording);
    // In a real implementation, this would integrate with Web Speech API
  };

  // Copy message to clipboard
  const copyMessage = (content: string, messageId: string) => {
    navigator.clipboard.writeText(content).then(() => {
      setCopiedMessageId(messageId);
      setTimeout(() => setCopiedMessageId(null), 2000);
    });
  };

  // Manufacturing-specific quick actions
  const quickActions = [
    {
      id: 'safety-check',
      label: 'Safety Check',
      prompt: 'Perform safety procedures check for current operation',
      icon: '🛡️'
    },
    {
      id: 'quality-guidelines',
      label: 'Quality Guidelines',
      prompt: 'Show quality control guidelines and inspection procedures',
      icon: '✅'
    },
    {
      id: 'maintenance-help',
      label: 'Maintenance Help',
      prompt: 'Provide maintenance guidance and troubleshooting steps',
      icon: '🔧'
    },
    {
      id: 'technical-specs',
      label: 'Technical Specs',
      prompt: 'Explain technical specifications and requirements',
      icon: '📋'
    }
  ];

  // Adjust input height based on content
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.style.height = 'auto';
      inputRef.current.style.height = `${Math.min(inputRef.current.scrollHeight, 120)}px`;
    }
  }, [input]);

  return (
    <div className={cn("flex flex-col h-full bg-background", className)}>
      {/* Messages Container */}
      <div className="flex-1 overflow-auto p-4 space-y-4">
        <AnimatePresence initial={false}>
          {messages.map((message) => (
            <motion.div
              key={message.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className={cn(
                "flex gap-3 max-w-4xl",
                message.role === 'user' ? "ml-auto" : "mr-auto"
              )}
            >
              {/* Avatar */}
              <div className={cn(
                "w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0",
                message.role === 'user'
                  ? "bg-primary text-primary-foreground"
                  : message.role === 'assistant'
                  ? "bg-gradient-to-br from-blue-500 to-purple-600 text-white"
                  : "bg-muted text-muted-foreground"
              )}>
                {message.role === 'user' ? (
                  <User className="w-4 h-4" />
                ) : message.role === 'assistant' ? (
                  <Bot className="w-4 h-4" />
                ) : (
                  <Sparkles className="w-4 h-4" />
                )}
              </div>

              {/* Message Content */}
              <div className={cn(
                "flex-1 p-4 rounded-lg",
                message.role === 'user'
                  ? "bg-primary text-primary-foreground ml-auto"
                  : message.role === 'assistant'
                  ? "bg-gradient-to-br from-blue-50 to-purple-50 dark:from-blue-950/20 dark:to-purple-950/20 border border-blue-200 dark:border-blue-800"
                  : "bg-muted text-muted-foreground"
              )}>
                {/* Message Header */}
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-medium opacity-70">
                    {message.role === 'user' ? 'You' : message.role === 'assistant' ? 'AI Assistant' : 'System'}
                  </span>
                  <span className="text-xs opacity-50">
                    {new Date(message.timestamp).toLocaleTimeString()}
                  </span>
                </div>

                {/* Message Text */}
                <div className="prose prose-sm max-w-none dark:prose-invert">
                  {message.isStreaming ? (
                    <span className="flex items-center gap-1">
                      <span>{message.content}</span>
                      <motion.div
                        animate={{ opacity: [0, 1, 0] }}
                        transition={{ repeat: Infinity, duration: 1.5 }}
                        className="inline-block w-2 h-4 bg-current ml-1"
                      />
                    </span>
                  ) : (
                    <span className="whitespace-pre-wrap">{message.content}</span>
                  )}
                </div>

                {/* Sources */}
                {message.sources && message.sources.length > 0 && (
                  <div className="mt-3 pt-3 border-t border-current/10">
                    <div className="text-xs font-medium mb-2 opacity-70">Sources:</div>
                    <div className="flex flex-wrap gap-1">
                      {message.sources.map((source, index) => (
                        <span
                          key={index}
                          className="inline-flex items-center px-2 py-1 rounded text-xs bg-current/10"
                        >
                          {source.title}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Quick Actions */}
                {message.quickActions && message.quickActions.length > 0 && (
                  <div className="mt-3 pt-3 border-t border-current/10">
                    <div className="flex gap-2">
                      {message.quickActions.map((action, index) => (
                        <Button
                          key={index}
                          variant="ghost"
                          size="sm"
                          className="h-6 px-2 text-xs"
                          onClick={() => onSendMessage(action)}
                        >
                          {action}
                        </Button>
                      ))}
                    </div>
                  </div>
                )}

                {/* Message Actions */}
                {message.role !== 'system' && (
                  <div className="flex gap-1 mt-2">
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-6 w-6 p-0 opacity-50 hover:opacity-100"
                      onClick={() => copyMessage(message.content, message.id)}
                    >
                      {copiedMessageId === message.id ? (
                        <Check className="w-3 h-3" />
                      ) : (
                        <Copy className="w-3 h-3" />
                      )}
                    </Button>
                  </div>
                )}
              </div>
            </motion.div>
          ))}
        </AnimatePresence>

        {/* Loading Indicator */}
        {isLoading && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex gap-3 max-w-4xl"
          >
            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 text-white flex items-center justify-center">
              <Bot className="w-4 h-4" />
            </div>
            <div className="bg-gradient-to-br from-blue-50 to-purple-50 dark:from-blue-950/20 dark:to-purple-950/20 border border-blue-200 dark:border-blue-800 p-4 rounded-lg">
              <div className="flex items-center gap-2">
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ repeat: Infinity, duration: 1, ease: 'linear' }}
                >
                  <RefreshCw className="w-4 h-4 text-blue-600" />
                </motion.div>
                <span className="text-sm text-blue-600">AI Assistant is thinking...</span>
              </div>
            </div>
          </motion.div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Quick Actions Bar */}
      {showQuickActions && messages.length === 0 && (
        <div className="border-t border-border p-4">
          <div className="mb-3">
            <h4 className="text-sm font-medium text-muted-foreground mb-2">Quick Actions</h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
              {quickActions.map((action) => (
                <Button
                  key={action.id}
                  variant="outline"
                  size="sm"
                  onClick={() => onSendMessage(action.prompt)}
                  className="h-auto p-3 flex flex-col items-start gap-1"
                >
                  <span className="text-lg">{action.icon}</span>
                  <span className="text-xs font-medium">{action.label}</span>
                </Button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Input Area */}
      <div className="border-t border-border p-4">
        <form onSubmit={handleSubmit} className="space-y-2">
          <div className="flex gap-2">
            {/* File Upload Button */}
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="shrink-0"
              title="Upload file"
            >
              <Paperclip className="w-4 h-4" />
            </Button>

            {/* Input Field */}
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSubmit();
                }
              }}
              placeholder={placeholder}
              className="flex-1 resize-none border border-border rounded-lg px-3 py-2 bg-background focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent min-h-[40px] max-h-[120px]"
              rows={1}
              disabled={isLoading}
            />

            {/* Voice Recording Button */}
            <Button
              type="button"
              variant={isRecording ? "destructive" : "outline"}
              size="sm"
              className="shrink-0"
              onClick={handleVoiceRecording}
              title={isRecording ? "Stop recording" : "Start voice recording"}
            >
              {isRecording ? <MicOff className="w-4 h-4" /> : <Mic className="w-4 h-4" />}
            </Button>

            {/* Send Button */}
            <Button
              type="submit"
              variant="quickAction"
              size="sm"
              disabled={!input.trim() || isLoading}
              className="shrink-0"
            >
              <Send className="w-4 h-4" />
            </Button>
          </div>

          {/* Manufacturing Context Indicator */}
          {manufacturingContext && (
            <div className="text-xs text-muted-foreground flex items-center gap-2">
              <span>Context:</span>
              <span className="font-medium">
                {manufacturingContext.equipment_type || 'General'} •
                {manufacturingContext.user_role || 'Operator'} •
                {manufacturingContext.facility_id || 'Default'}
              </span>
            </div>
          )}
        </form>
      </div>
    </div>
  );
};

export default Chat;