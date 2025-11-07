/**
 * WebSocket client for real-time interactions with the Knowledge Base API.
 *
 * Provides WebSocket connections for real-time search, suggestions,
 * and live updates with automatic reconnection and error handling.
 */

import WebSocket from 'ws';
import { EventEmitter } from 'events';
import { WebSocketError } from './errors';
import type {
  WebSocketMessage,
  WebSocketSearchMessage,
  WebSocketSearchResultsMessage,
  WebSocketProgressMessage,
  WebSocketSuggestionsMessage,
  WebSocketErrorMessage,
  WebSocketEventHandlers,
  WebSocketClientConfig,
  SearchResult,
  SearchAnalytics,
  EventHandler
} from './types';

// ============================================================================
// WebSocket Client Configuration
// ============================================================================

interface WebSocketConnection {
  ws: WebSocket | null;
  connected: boolean;
  reconnecting: boolean;
  url: string;
  reconnectAttempts: number;
  lastError?: Error;
}

// ============================================================================
// WebSocket Client Class
// ============================================================================

export class WebSocketClient extends EventEmitter {
  private readonly config: Required<WebSocketClientConfig>;
  private readonly connections: Map<'search' | 'personalized', WebSocketConnection>;
  private readonly heartbeatIntervals: Map<'search' | 'personalized', NodeJS.Timeout>;
  private running = false;

  constructor(config: WebSocketClientConfig = {}) {
    super();

    this.config = {
      baseURL: config.baseURL || 'ws://localhost:8006',
      personalizedURL: config.personalizedURL || 'ws://localhost:8007',
      apiKey: config.apiKey || '',
      pingInterval: config.pingInterval || 20000,
      pingTimeout: config.pingTimeout || 20000,
      closeTimeout: config.closeTimeout || 10000,
      maxReconnectAttempts: config.maxReconnectAttempts || 5,
      reconnectDelay: config.reconnectDelay || 1000,
    };

    this.connections = new Map([
      ['search', this.createConnection('search', this.config.baseURL + '/ws/search')],
      ['personalized', this.createConnection('personalized', this.config.personalizedURL + '/ws/personalized-search')]
    ]);

    this.heartbeatIntervals = new Map();
  }

  /**
   * Register event handler.
   */
  public on<T = unknown>(event: string, handler: EventHandler<T>): this {
    return super.on(event, handler);
  }

  /**
   * Register one-time event handler.
   */
  public once<T = unknown>(event: string, handler: EventHandler<T>): this {
    return super.once(event, handler);
  }

  /**
   * Remove event handler.
   */
  public off<T = unknown>(event: string, handler: EventHandler<T>): this {
    return super.off(event, handler);
  }

  /**
   * Connect to search WebSocket.
   */
  public async connectSearch(): Promise<void> {
    const connection = this.connections.get('search')!;
    if (connection.connected || connection.reconnecting) {
      return;
    }

    await this.connect('search');
  }

  /**
   * Connect to personalized search WebSocket.
   */
  public async connectPersonalized(): Promise<void> {
    const connection = this.connections.get('personalized')!;
    if (connection.connected || connection.reconnecting) {
      return;
    }

    await this.connect('personalized');
  }

  /**
   * Start WebSocket client and connect to both endpoints.
   */
  public async start(): Promise<void> {
    this.running = true;

    // Connect to both endpoints
    await Promise.all([
      this.connectSearch().catch(error => this.emit('error', error)),
      this.connectPersonalized().catch(error => this.emit('error', error))
    ]);
  }

  /**
   * Stop WebSocket client and close connections.
   */
  public async stop(): Promise<void> {
    this.running = false;

    // Clear heartbeat intervals
    for (const interval of this.heartbeatIntervals.values()) {
      clearInterval(interval);
    }
    this.heartbeatIntervals.clear();

    // Close connections
    const closePromises: Promise<void>[] = [];
    for (const [type, connection] of this.connections) {
      if (connection.ws) {
        closePromises.push(this.closeConnection(type, connection));
      }
    }

    await Promise.all(closePromises);
  }

  /**
   * Send search request via WebSocket.
   */
  public async sendSearchRequest(
    query: string,
    options: {
      strategy?: string;
      topK?: number;
      threshold?: number;
      connectionType?: 'search' | 'personalized';
    } = {}
  ): Promise<void> {
    const connectionType = options.connectionType || 'search';
    const connection = this.connections.get(connectionType)!;

    if (!connection.connected) {
      if (connectionType === 'search') {
        await this.connectSearch();
      } else {
        await this.connectPersonalized();
      }
    }

    const message: WebSocketSearchMessage = {
      type: 'search',
      query,
      strategy: options.strategy || 'unified',
      topK: options.topK || 10,
      threshold: options.threshold || 0.7,
      timestamp: new Date().toISOString()
    };

    await this.sendMessage(connectionType, message);
  }

  /**
   * Send suggestion request via WebSocket.
   */
  public async sendSuggestionRequest(
    query: string,
    options: {
      userId?: string;
      maxSuggestions?: number;
    } = {}
  ): Promise<void> {
    const connection = this.connections.get('personalized')!;

    if (!connection.connected) {
      await this.connectPersonalized();
    }

    const message = {
      type: 'suggestions',
      query,
      userId: options.userId,
      maxSuggestions: options.maxSuggestions || 5,
      timestamp: new Date().toISOString()
    };

    await this.sendMessage('personalized', message);
  }

  /**
   * Create connection configuration.
   */
  private createConnection(type: 'search' | 'personalized', url: string): WebSocketConnection {
    return {
      ws: null,
      connected: false,
      reconnecting: false,
      url,
      reconnectAttempts: 0
    };
  }

  /**
   * Connect to a specific WebSocket endpoint.
   */
  private async connect(type: 'search' | 'personalized'): Promise<void> {
    const connection = this.connections.get(type)!;
    if (connection.connected || connection.reconnecting) {
      return;
    }

    connection.reconnecting = true;
    connection.reconnectAttempts = 0;

    while (connection.reconnectAttempts < this.config.maxReconnectAttempts && this.running) {
      try {
        await this.connectOnce(type);
        return; // Connection successful
      } catch (error) {
        connection.lastError = error as Error;
        connection.reconnectAttempts++;

        if (connection.reconnectAttempts < this.config.maxReconnectAttempts && this.running) {
          const delay = this.config.reconnectDelay * Math.pow(2, connection.reconnectAttempts - 1);
          await this.sleep(delay);
        }
      }
    }

    connection.reconnecting = false;
    const error = new WebSocketError(
      `Failed to connect to ${type} WebSocket after ${connection.reconnectAttempts} attempts`
    );
    this.emit('error', error);
    throw error;
  }

  /**
   * Perform a single connection attempt.
   */
  private async connectOnce(type: 'search' | 'personalized'): Promise<void> {
    const connection = this.connections.get(type)!;

    return new Promise((resolve, reject) => {
      const headers: Record<string, string> = {};
      if (this.config.apiKey) {
        headers['Authorization'] = `Bearer ${this.config.apiKey}`;
      }

      const ws = new WebSocket(connection.url, { headers });

      // Connection timeout
      const timeout = setTimeout(() => {
        ws.terminate();
        reject(new WebSocketError(`Connection timeout for ${type} WebSocket`));
      }, this.config.closeTimeout);

      ws.on('open', () => {
        clearTimeout(timeout);
        connection.ws = ws;
        connection.connected = true;
        connection.reconnecting = false;
        connection.reconnectAttempts = 0;

        this.emit('open', type);
        this.setupHeartbeat(type);
        this.setupMessageHandlers(type, ws);
        resolve();
      });

      ws.on('error', (error) => {
        clearTimeout(timeout);
        reject(new WebSocketError(`WebSocket error: ${error.message}`, { code: ws.readyState, reason: error.message }));
      });
    });
  }

  /**
   * Setup message handlers for a WebSocket connection.
   */
  private setupMessageHandlers(type: 'search' | 'personalized', ws: WebSocket): void {
    ws.on('message', (data: WebSocket.Data) => {
      try {
        const message = JSON.parse(data.toString()) as WebSocketMessage;
        this.handleMessage(message, type);
      } catch (error) {
        const wsError = new WebSocketError(`Invalid JSON message: ${data.toString()}`);
        this.emit('searchError', wsError);
      }
    });

    ws.on('close', (code: number, reason: string) => {
      const connection = this.connections.get(type)!;
      connection.connected = false;
      connection.ws = null;

      // Clear heartbeat
      const interval = this.heartbeatIntervals.get(type);
      if (interval) {
        clearInterval(interval);
        this.heartbeatIntervals.delete(type);
      }

      this.emit('close', type);

      // Attempt reconnection if still running
      if (this.running && code !== 1000) {
        setTimeout(() => {
          this.connect(type).catch(error => this.emit('error', error));
        }, this.config.reconnectDelay);
      }
    });

    ws.on('pong', () => {
      // Handle pong response
      const connection = this.connections.get(type)!;
      connection.lastError = undefined;
    });
  }

  /**
   * Setup heartbeat for a connection.
   */
  private setupHeartbeat(type: 'search' | 'personalized'): void {
    const interval = setInterval(() => {
      const connection = this.connections.get(type)!;
      if (connection.ws && connection.ws.readyState === WebSocket.OPEN) {
        connection.ws.ping();
      }
    }, this.config.pingInterval);

    this.heartbeatIntervals.set(type, interval);
  }

  /**
   * Handle incoming WebSocket message.
   */
  private handleMessage(message: WebSocketMessage, connectionType: string): void {
    this.emit('message', message, connectionType);

    switch (message.type) {
      case 'results':
        this.handleSearchResults(message as WebSocketSearchResultsMessage);
        break;
      case 'progress':
        this.handleProgressMessage(message as WebSocketProgressMessage);
        break;
      case 'suggestions':
        this.handleSuggestionsMessage(message as WebSocketSuggestionsMessage);
        break;
      case 'error':
        this.handleErrorMessage(message as WebSocketErrorMessage);
        break;
    }
  }

  /**
   * Handle search results message.
   */
  private handleSearchResults(message: WebSocketSearchResultsMessage): void {
    this.emit('results', {
      searchId: message.searchId,
      results: message.results,
      executionTime: message.executionTime,
      analytics: message.analytics,
      timestamp: new Date()
    });
  }

  /**
   * Handle progress message.
   */
  private handleProgressMessage(message: WebSocketProgressMessage): void {
    this.emit('progress', {
      searchId: message.searchId,
      status: message.status,
      message: message.message,
      timestamp: new Date()
    });
  }

  /**
   * Handle suggestions message.
   */
  private handleSuggestionsMessage(message: WebSocketSuggestionsMessage): void {
    this.emit('suggestions', {
      query: message.query,
      suggestions: message.suggestions,
      personalized: message.personalized,
      timestamp: new Date()
    });
  }

  /**
   * Handle error message.
   */
  private handleErrorMessage(message: WebSocketErrorMessage): void {
    const error = new WebSocketError(message.message, { code: parseInt(message.code || '0') });
    this.emit('searchError', error);
  }

  /**
   * Send message through WebSocket.
   */
  private async sendMessage(type: 'search' | 'personalized', message: unknown): Promise<void> {
    const connection = this.connections.get(type)!;

    if (!connection.ws || connection.ws.readyState !== WebSocket.OPEN) {
      throw new WebSocketError(`WebSocket ${type} is not connected`);
    }

    const messageStr = JSON.stringify(message);

    return new Promise((resolve, reject) => {
      connection.ws!.send(messageStr, (error?: Error) => {
        if (error) {
          reject(new WebSocketError(`Failed to send message: ${error.message}`));
        } else {
          resolve();
        }
      });
    });
  }

  /**
   * Close WebSocket connection.
   */
  private async closeConnection(type: 'search' | 'personalized', connection: WebSocketConnection): Promise<void> {
    return new Promise((resolve) => {
      if (connection.ws) {
        connection.ws.close(1000, 'Client closing');
        connection.ws.on('close', () => resolve());

        // Force close after timeout
        setTimeout(() => {
          if (connection.ws) {
            connection.ws.terminate();
          }
          resolve();
        }, this.config.closeTimeout);
      } else {
        resolve();
      }
    });
  }

  /**
   * Sleep for a specified number of milliseconds.
   */
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/**
 * Create a WebSocket client for search operations.
 */
export function createSearchWebSocket(config?: WebSocketClientConfig): WebSocketClient {
  return new WebSocketClient(config);
}

/**
 * Create a WebSocket client for personalized search operations.
 */
export function createPersonalizedWebSocket(config?: WebSocketClientConfig): WebSocketClient {
  return new WebSocketClient(config);
}