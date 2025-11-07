import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Send,
  Paperclip,
  Mic,
  MicOff,
  Settings,
  Trash2,
  Copy,
  Download,
  RefreshCw,
  ChevronDown,
  User,
  Bot,
  Lightbulb,
  BookOpen,
  Cpu
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { useThemeStore } from '@/store/theme';
import { api } from '@/services/api';
import type { ChatMessage, ChatSession, ManufacturingSystem } from '@/types';
import { cn, formatRelativeTime, copyToClipboard } from '@/lib/utils';

interface ChatInterfaceProps {
  systemId?: string;
  className?: string;
}

export function ChatInterface({ systemId, className }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [sessionId, setSessionId] = useState<string>('');
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [relatedSystems, setRelatedSystems] = useState<ManufacturingSystem[]>([]);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const { resolvedTheme } = useThemeStore();

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  // Load suggestions on mount
  useEffect(() => {
    loadSuggestions();
  }, []);

  const loadSuggestions = async () => {
    try {
      const suggestionsData = await api.chat.getSuggestions();
      setSuggestions(suggestionsData);
    } catch (error) {
      console.error('Failed to load suggestions:', error);
    }
  };

  const handleSend = async (messageContent: string = input) => {
    if (!messageContent.trim()) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: messageContent,
      timestamp: new Date().toISOString(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await api.chat.sendMessage({
        message: messageContent,
        sessionId: sessionId || undefined,
        systemId,
        context: messages.slice(-5).map(m => m.content),
      });

      const assistantMessage: ChatMessage = {
        id: response.message.id,
        role: 'assistant',
        content: response.message.content,
        timestamp: response.message.timestamp,
        metadata: response.message.metadata,
      };

      setMessages(prev => [...prev, assistantMessage]);
      setSessionId(response.sessionId);

      // Update suggestions if provided
      if (response.suggestions) {
        setSuggestions(response.suggestions);
      }
    } catch (error) {
      console.error('Failed to send message:', error);

      const errorMessage: ChatMessage = {
        id: Date.now().toString(),
        role: 'assistant',
        content: 'I apologize, but I encountered an error while processing your request. Please try again.',
        timestamp: new Date().toISOString(),
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleClearChat = () => {
    setMessages([]);
    setSessionId('');
    inputRef.current?.focus();
  };

  const handleCopyMessage = async (message: ChatMessage) => {
    await copyToClipboard(message.content);
  };

  const handleSuggestionClick = (suggestion: string) => {
    handleSend(suggestion);
  };

  const startRecording = () => {
    setIsRecording(true);
    // TODO: Implement voice recording
    console.log('Starting voice recording...');
  };

  const stopRecording = () => {
    setIsRecording(false);
    // TODO: Implement voice recording
    console.log('Stopping voice recording...');
  };

  const getMessageIcon = (role: string) => {
    switch (role) {
      case 'user':
        return <User className="h-4 w-4" />;
      case 'assistant':
        return <Bot className="h-4 w-4" />;
      default:
        return <Cpu className="h-4 w-4" />;
    }
  };

  const getSuggestionIcon = (suggestion: string) => {
    if (suggestion.toLowerCase().includes('help') || suggestion.toLowerCase().includes('assist')) {
      return <Lightbulb className="h-4 w-4" />;
    }
    if (suggestion.toLowerCase().includes('search') || suggestion.toLowerCase().includes('find')) {
      return <BookOpen className="h-4 w-4" />;
    }
    return <Cpu className="h-4 w-4" />;
  };

  return (
    <div className={cn("flex flex-col h-full", className)}>
      {/* Chat Header */}
      <CardHeader className="border-b px-4 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="flex items-center space-x-2">
              <Cpu className="h-5 w-5 text-primary" />
              <h3 className="font-semibold">Manufacturing Assistant</h3>
            </div>
            {systemId && (
              <span className="text-sm text-muted-foreground">
                System: {systemId}
              </span>
            )}
          </div>
          <div className="flex items-center space-x-2">
            <Button
              variant="ghost"
              size="icon"
              onClick={handleClearChat}
              className="h-8 w-8"
            >
              <Trash2 className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8"
            >
              <Settings className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </CardHeader>

      {/* Messages Area */}
      <CardContent className="flex-1 overflow-y-auto p-4 space-y-4">
        <AnimatePresence initial={false}>
          {messages.map((message) => (
            <motion.div
              key={message.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
              className={cn(
                "flex items-start space-x-3 group",
                message.role === 'user' ? "justify-end" : "justify-start"
              )}
            >
              {message.role !== 'user' && (
                <div className={cn(
                  "flex items-center justify-center w-8 h-8 rounded-full flex-shrink-0",
                  message.role === 'assistant'
                    ? "bg-primary/10 text-primary"
                    : "bg-muted text-muted-foreground"
                )}>
                  {getMessageIcon(message.role)}
                </div>
              )}

              <div className={cn(
                "max-w-[80%] rounded-lg px-4 py-3 relative group",
                message.role === 'user'
                  ? "bg-primary text-primary-foreground ml-auto"
                  : "bg-muted"
              )}>
                <div className="whitespace-pre-wrap text-sm leading-relaxed">
                  {message.content}
                </div>

                {/* Message metadata */}
                {message.metadata && (
                  <div className="mt-2 pt-2 border-t border-current/20">
                    <div className="flex items-center justify-between text-xs opacity-70">
                      <span>
                        {message.metadata.model && `Model: ${message.metadata.model}`}
                        {message.metadata.responseTime && ` • ${message.metadata.responseTime}ms`}
                        {message.metadata.tokens && ` • ${message.metadata.tokens} tokens`}
                      </span>
                      <span>{formatRelativeTime(message.timestamp)}</span>
                    </div>
                  </div>
                )}

                {/* Action buttons */}
                <div className="absolute -top-2 -right-2 opacity-0 group-hover:opacity-100 transition-opacity">
                  <div className="flex space-x-1">
                    <Button
                      variant="outline"
                      size="icon"
                      onClick={() => handleCopyMessage(message)}
                      className="h-6 w-6 bg-background"
                    >
                      <Copy className="h-3 w-3" />
                    </Button>
                  </div>
                </div>
              </div>

              {message.role === 'user' && (
                <div className="flex items-center justify-center w-8 h-8 rounded-full bg-primary/10 text-primary flex-shrink-0">
                  {getMessageIcon(message.role)}
                </div>
              )}
            </motion.div>
          ))}
        </AnimatePresence>

        {/* Loading indicator */}
        {isLoading && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex items-center space-x-3"
          >
            <div className="flex items-center justify-center w-8 h-8 rounded-full bg-primary/10 text-primary">
              <Bot className="h-4 w-4" />
            </div>
            <div className="bg-muted rounded-lg px-4 py-3">
              <div className="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </motion.div>
        )}

        <div ref={messagesEndRef} />
      </CardContent>

      {/* Suggestions */}
      {suggestions.length > 0 && messages.length === 0 && (
        <div className="px-4 pb-2">
          <div className="flex flex-wrap gap-2">
            {suggestions.slice(0, 4).map((suggestion, index) => (
              <Button
                key={index}
                variant="outline"
                size="sm"
                onClick={() => handleSuggestionClick(suggestion)}
                className="h-8 text-xs"
              >
                {getSuggestionIcon(suggestion)}
                <span className="ml-2">{suggestion}</span>
              </Button>
            ))}
          </div>
        </div>
      )}

      {/* Input Area */}
      <div className="border-t p-4">
        <div className="flex items-end space-x-2">
          <Button
            variant="ghost"
            size="icon"
            className="h-10 w-10 flex-shrink-0"
          >
            <Paperclip className="h-4 w-4" />
          </Button>

          <div className="flex-1 relative">
            <Input
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask about manufacturing systems, processes, or get assistance..."
              disabled={isLoading}
              className="pr-10 min-h-[40px]"
            />
            <div className="absolute right-2 top-1/2 -translate-y-1/2">
              <Button
                variant="ghost"
                size="icon"
                onMouseDown={startRecording}
                onMouseUp={stopRecording}
                className={cn(
                  "h-6 w-6",
                  isRecording && "text-destructive animate-pulse"
                )}
              >
                {isRecording ? <MicOff className="h-3 w-3" /> : <Mic className="h-3 w-3" />}
              </Button>
            </div>
          </div>

          <Button
            onClick={() => handleSend()}
            disabled={!input.trim() || isLoading}
            size="icon"
            className="h-10 w-10 flex-shrink-0"
          >
            <Send className="h-4 w-4" />
          </Button>
        </div>

        <div className="mt-2 text-xs text-muted-foreground text-center">
          Press Enter to send, Shift+Enter for new line
        </div>
      </div>
    </div>
  );
}

export default ChatInterface;