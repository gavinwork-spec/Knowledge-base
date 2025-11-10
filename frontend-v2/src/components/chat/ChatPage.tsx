import React, { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { MessageSquare, Bot, User, Settings, History, Plus } from 'lucide-react';

// Import components
import Chat from '../ui/chat';

// Import services and stores
import apiService from '@lib/api';
import { useChatStore, useManufacturingStore, useUIStore } from '@stores';
import { ChatMessage, ChatSession } from '@types';

// Import UI components
import Button from '../ui/button';
import { cn } from '@lib/utils';

const ChatPage: React.FC = () => {
  const { context } = useManufacturingStore();
  const {
    currentSession,
    sessions,
    setCurrentSession,
    addSession,
    updateSession,
    addMessage,
    quickActions,
    setQuickActions,
    isLoading,
    setLoading,
    setError
  } = useChatStore();

  const { addNotification } = useUIStore();
  const [ws, setWs] = useState<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Initialize WebSocket connection
  useEffect(() => {
    const connectWebSocket = () => {
      try {
        const websocket = apiService.connectWebSocket('chat');
        setWs(websocket);

        websocket.onopen = () => {
          setIsConnected(true);
          console.log('WebSocket connected');
          addNotification({
            type: 'success',
            title: 'Connected',
            message: 'Chat service is now connected',
          });
        };

        websocket.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            handleWebSocketMessage(data);
          } catch (error) {
            console.error('Error parsing WebSocket message:', error);
          }
        };

        websocket.onclose = () => {
          setIsConnected(false);
          console.log('WebSocket disconnected');
          // Attempt to reconnect after 3 seconds
          setTimeout(connectWebSocket, 3000);
        };

        websocket.onerror = (error) => {
          console.error('WebSocket error:', error);
          setIsConnected(false);
          addNotification({
            type: 'error',
            title: 'Connection Error',
            message: 'Failed to connect to chat service',
          });
        };
      } catch (error) {
        console.error('Error creating WebSocket connection:', error);
        addNotification({
          type: 'error',
          title: 'Connection Error',
          message: 'Unable to establish chat connection',
        });
      }
    };

    connectWebSocket();

    return () => {
      if (ws) {
        ws.close();
      }
    };
  }, []);

  // Handle WebSocket messages
  const handleWebSocketMessage = (data: any) => {
    switch (data.type) {
      case 'message':
        if (currentSession) {
          const message: ChatMessage = {
            id: data.message.id,
            role: data.message.role,
            content: data.message.content,
            timestamp: data.message.timestamp,
            manufacturing_context: data.message.manufacturing_context,
            metadata: data.message.metadata
          };
          addMessage(currentSession.id, message);
        }
        break;

      case 'typing':
        // Handle typing indicators
        break;

      case 'quick_actions':
        if (data.quick_actions) {
          setQuickActions(data.quick_actions);
        }
        break;

      case 'error':
        setError(data.message);
        addNotification({
          type: 'error',
          title: 'Chat Error',
          message: data.message,
        });
        break;

      default:
        console.log('Unknown WebSocket message type:', data.type);
    }
  };

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [currentSession?.messages]);

  // Create new chat session
  const createNewSession = async () => {
    setLoading(true);
    try {
      const response = await apiService.createSession(
        'New Chat Session',
        context
      );

      if (response.success && response.data) {
        const newSession = response.data as ChatSession;
        addSession(newSession);
        setCurrentSession(newSession);

        // Send initial greeting
        if (ws && isConnected) {
          ws.send(JSON.stringify({
            type: 'session_created',
            session_id: newSession.id,
            context: context
          }));
        }
      }
    } catch (error) {
      console.error('Error creating session:', error);
      addNotification({
        type: 'error',
        title: 'Session Error',
        message: 'Failed to create new chat session',
      });
    } finally {
      setLoading(false);
    }
  };

  // Handle message sending
  const handleSendMessage = async (message: string) => {
    if (!message.trim() || !currentSession || !ws) return;

    // Add user message immediately for better UX
    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: message.trim(),
      timestamp: new Date().toISOString(),
      manufacturing_context: context
    };
    addMessage(currentSession.id, userMessage);

    // Send message via WebSocket
    ws.send(JSON.stringify({
      type: 'message',
      session_id: currentSession.id,
      message: message.trim(),
      context: context
    }));
  };

  // Switch to a different session
  const switchSession = (session: ChatSession) => {
    setCurrentSession(session);

    if (ws && isConnected) {
      ws.send(JSON.stringify({
        type: 'session_switched',
        session_id: session.id,
        context: context
      }));
    }
  };

  // Delete a session
  const deleteSession = async (sessionId: string) => {
    try {
      await apiService.deleteSession(sessionId);

      // If deleting current session, create a new one
      if (currentSession?.id === sessionId) {
        await createNewSession();
      }
    } catch (error) {
      console.error('Error deleting session:', error);
      addNotification({
        type: 'error',
        title: 'Delete Error',
        message: 'Failed to delete chat session',
      });
    }
  };

  // Manufacturing-specific quick actions
  const manufacturingQuickActions = [
    {
      id: 'safety-check',
      title: 'Safety Check',
      description: 'Perform comprehensive safety procedures check',
      template: 'Please perform a comprehensive safety check for my current operation. Include:\n- Required PPE and safety equipment\n- Machine safety guards and interlocks\n- Emergency procedures\n- Lockout/tagout requirements\n- Relevant OSHA and ANSI standards',
      equipment_types: ['cnc_milling', 'cnc_turning', 'grinding'],
      process_types: ['machining', 'setup']
    },
    {
      id: 'quality-inspection',
      title: 'Quality Inspection',
      description: 'Generate quality inspection procedures',
      template: 'Please provide quality inspection procedures for:\n- First article inspection\n- In-process quality checks\n- Final inspection requirements\n- Measurement equipment needed\n- Acceptance criteria and tolerances\n- Documentation requirements',
      equipment_types: ['measurement', 'inspection'],
      process_types: ['quality_control', 'inspection']
    },
    {
      id: 'maintenance-help',
      title: 'Maintenance Help',
      description: 'Get maintenance guidance and troubleshooting',
      template: 'I need maintenance assistance for:\n- Troubleshooting common issues\n- Preventive maintenance schedule\n- Parts replacement guidelines\n- Lubrication and cleaning procedures\n- Safety considerations during maintenance',
      equipment_types: ['cnc_milling', 'cnc_turning', 'grinding', 'assembly'],
      process_types: ['maintenance']
    },
    {
      id: 'technical-specs',
      title: 'Technical Specifications',
      description: 'Explain technical specifications',
      template: 'Please explain the technical specifications for:\n- Equipment capabilities and limitations\n- Material requirements and tolerances\n- Process parameters and settings\n- Tooling requirements\n- Safety and operational considerations',
      equipment_types: ['cnc_milling', 'cnc_turning', 'grinding', 'measurement'],
      process_types: ['machining']
    }
  ];

  // Filter quick actions based on manufacturing context
  const filteredQuickActions = manufacturingQuickActions.filter(action => {
    if (context.equipment_type && action.equipment_types) {
      return action.equipment_types.includes(context.equipment_type);
    }
    if (context.process_type && action.process_types) {
      return action.process_types.includes(context.process_type);
    }
    return true;
  });

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="h-full flex"
    >
      {/* Sidebar with Sessions */}
      <div className="w-80 border-r border-border flex flex-col">
        {/* Sidebar Header */}
        <div className="p-4 border-b border-border">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold flex items-center gap-2">
              <MessageSquare className="w-5 h-5" />
              Chat Sessions
            </h2>
            <div className="flex items-center gap-1">
              <div className={cn(
                "w-2 h-2 rounded-full",
                isConnected ? "bg-green-500" : "bg-red-500"
              )} />
              <span className="text-xs text-muted-foreground">
                {isConnected ? 'Connected' : 'Disconnected'}
              </span>
            </div>
          </div>

          <Button
            onClick={createNewSession}
            className="w-full"
            disabled={isLoading}
          >
            <Plus className="w-4 h-4 mr-2" />
            New Conversation
          </Button>
        </div>

        {/* Sessions List */}
        <div className="flex-1 overflow-auto p-4">
          <div className="space-y-2">
            {sessions.map((session) => (
              <button
                key={session.id}
                onClick={() => switchSession(session)}
                className={cn(
                  "w-full text-left p-3 rounded-lg border transition-colors hover:bg-accent",
                  currentSession?.id === session.id
                    ? "border-primary bg-primary/5"
                    : "border-border"
                )}
              >
                <div className="font-medium text-sm truncate">
                  {session.title}
                </div>
                <div className="text-xs text-muted-foreground mt-1">
                  {session.messages.length} messages •
                  {new Date(session.updated_at).toLocaleDateString()}
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Manufacturing Context */}
        {context && (
          <div className="p-4 border-t border-border">
            <h3 className="text-sm font-medium mb-2">Manufacturing Context</h3>
            <div className="space-y-1 text-xs text-muted-foreground">
              <div>Equipment: {context.equipment_type || 'General'}</div>
              <div>Role: {context.user_role || 'Operator'}</div>
              <div>Facility: {context.facility_id || 'Default'}</div>
            </div>
          </div>
        )}
      </div>

      {/* Chat Interface */}
      <div className="flex-1 flex flex-col">
        {currentSession ? (
          <>
            {/* Session Header */}
            <div className="border-b border-border p-4">
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-semibold">
                  {currentSession.title}
                </h2>
                <div className="flex items-center gap-2">
                  <Button variant="ghost" size="sm">
                    <Settings className="w-4 h-4" />
                  </Button>
                  <Button variant="ghost" size="sm">
                    <History className="w-4 h-4" />
                  </Button>
                </div>
              </div>
            </div>

            {/* Chat Messages */}
            <div className="flex-1 overflow-hidden">
              <Chat
                messages={currentSession.messages}
                onSendMessage={handleSendMessage}
                isLoading={isLoading}
                showQuickActions={true}
                manufacturingContext={context}
                quickActions={filteredQuickActions}
              />
            </div>
          </>
        ) : (
          /* Welcome Screen */
          <div className="flex-1 flex items-center justify-center p-8">
            <div className="text-center max-w-md">
              <div className="mb-6">
                <Bot className="w-16 h-16 mx-auto text-primary mb-4" />
                <h2 className="text-2xl font-bold mb-2">Manufacturing AI Assistant</h2>
                <p className="text-muted-foreground">
                  Get instant help with safety procedures, quality guidelines, equipment troubleshooting, and technical specifications.
                </p>
              </div>

              <div className="space-y-4">
                <h3 className="font-medium">Quick Start:</h3>
                <div className="grid grid-cols-1 gap-2">
                  {filteredQuickActions.map((action) => (
                    <Button
                      key={action.id}
                      variant="outline"
                      className="h-auto p-4 text-left justify-start"
                      onClick={() => {
                        createNewSession().then(() => {
                          setTimeout(() => {
                            handleSendMessage(action.template);
                          }, 500);
                        });
                      }}
                    >
                      <div>
                        <div className="font-medium">{action.title}</div>
                        <div className="text-xs text-muted-foreground mt-1">
                          {action.description}
                        </div>
                      </div>
                    </Button>
                  ))}
                </div>

                <Button
                  onClick={createNewSession}
                  className="w-full"
                  disabled={isLoading}
                >
                  Start New Conversation
                </Button>
              </div>
            </div>
          </div>
        )}
      </div>

      <div ref={messagesEndRef} />
    </motion.div>
  );
};

export default ChatPage;