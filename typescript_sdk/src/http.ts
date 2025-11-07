/**
 * HTTP client utilities for the Knowledge Base API SDK.
 *
 * Provides a lightweight HTTP client with retry logic, error handling,
 * and request/response interceptors.
 */

import { KnowledgeBaseError, createExceptionFromResponse } from './errors';
import type { HTTPRequestConfig, HTTPResponse } from './types';

// ============================================================================
// HTTP Client Configuration
// ============================================================================

export interface HTTPClientConfig {
  baseURL?: string;
  timeout?: number;
  maxRetries?: number;
  retryDelay?: number;
  headers?: Record<string, string>;
  apiKey?: string;
}

// ============================================================================
// HTTP Client Class
// ============================================================================

export class HTTPClient {
  private readonly baseURL: string;
  private readonly timeout: number;
  private readonly maxRetries: number;
  private readonly retryDelay: number;
  private readonly defaultHeaders: Record<string, string>;

  constructor(config: HTTPClientConfig = {}) {
    this.baseURL = config.baseURL || '';
    this.timeout = config.timeout || 30000;
    this.maxRetries = config.maxRetries || 3;
    this.retryDelay = config.retryDelay || 1000;

    // Build default headers
    this.defaultHeaders = {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      'User-Agent': 'knowledge-base-sdk-ts/1.0.0',
      ...config.headers
    };

    // Add authorization header if API key is provided
    if (config.apiKey) {
      this.defaultHeaders['Authorization'] = `Bearer ${config.apiKey}`;
    }
  }

  /**
   * Make an HTTP request with retry logic and error handling.
   */
  public async request<T = unknown>(config: HTTPRequestConfig): Promise<HTTPResponse<T>> {
    const { method, url, data, params, headers = {}, timeout = this.timeout } = config;

    // Build full URL
    const fullURL = this.buildURL(url, params);

    // Merge headers
    const requestHeaders = { ...this.defaultHeaders, ...headers };

    // Build request options for Node.js fetch API
    const requestOptions: RequestInit = {
      method,
      headers: requestHeaders,
      signal: AbortSignal.timeout(timeout)
    };

    // Add body for POST/PUT/PATCH requests
    if (data && ['POST', 'PUT', 'PATCH'].includes(method.toUpperCase())) {
      requestOptions.body = typeof data === 'string' ? data : JSON.stringify(data);
    }

    let lastError: Error | null = null;

    // Retry logic
    for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
      try {
        const response = await fetch(fullURL, requestOptions);
        return await this.handleResponse<T>(response);
      } catch (error) {
        lastError = error as Error;

        // Don't retry on client errors (except 429)
        if (error instanceof KnowledgeBaseError) {
          if (error.status && error.status < 500 && error.status !== 429) {
            throw error;
          }
        }

        // If this is the last attempt, throw the error
        if (attempt === this.maxRetries) {
          break;
        }

        // Wait before retrying (exponential backoff)
        const delay = this.retryDelay * Math.pow(2, attempt);
        await this.sleep(delay);
      }
    }

    // All retries exhausted
    throw lastError || new KnowledgeBaseError('Request failed after all retries');
  }

  /**
   * Convenience method for GET requests.
   */
  public async get<T = unknown>(
    url: string,
    options: Omit<HTTPRequestConfig, 'method' | 'data'> = {}
  ): Promise<HTTPResponse<T>> {
    return this.request<T>({ ...options, method: 'GET', url });
  }

  /**
   * Convenience method for POST requests.
   */
  public async post<T = unknown>(
    url: string,
    data?: unknown,
    options: Omit<HTTPRequestConfig, 'method' | 'data'> = {}
  ): Promise<HTTPResponse<T>> {
    return this.request<T>({ ...options, method: 'POST', url, data });
  }

  /**
   * Convenience method for PUT requests.
   */
  public async put<T = unknown>(
    url: string,
    data?: unknown,
    options: Omit<HTTPRequestConfig, 'method' | 'data'> = {}
  ): Promise<HTTPResponse<T>> {
    return this.request<T>({ ...options, method: 'PUT', url, data });
  }

  /**
   * Convenience method for DELETE requests.
   */
  public async delete<T = unknown>(
    url: string,
    options: Omit<HTTPRequestConfig, 'method' | 'data'> = {}
  ): Promise<HTTPResponse<T>> {
    return this.request<T>({ ...options, method: 'DELETE', url });
  }

  /**
   * Build full URL with query parameters.
   */
  private buildURL(url: string, params?: Record<string, unknown>): string {
    let fullURL = url;

    // Add base URL if URL is relative
    if (!url.startsWith('http') && this.baseURL) {
      fullURL = `${this.baseURL}${url}`;
    }

    // Add query parameters
    if (params && Object.keys(params).length > 0) {
      const searchParams = new URLSearchParams();
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined && value !== null) {
          searchParams.append(key, String(value));
        }
      });

      const separator = fullURL.includes('?') ? '&' : '?';
      fullURL = `${fullURL}${separator}${searchParams.toString()}`;
    }

    return fullURL;
  }

  /**
   * Handle HTTP response and parse data.
   */
  private async handleResponse<T>(response: Response): Promise<HTTPResponse<T>> {
    const requestId = response.headers.get('X-Request-ID') || undefined;

    // Handle successful responses
    if (response.ok) {
      try {
        const data = await response.json();
        return {
          data,
          status: response.status,
          statusText: response.statusText,
          headers: this.parseHeaders(response.headers)
        };
      } catch (error) {
        // Return raw text if JSON parsing fails
        const text = await response.text();
        return {
          data: text as unknown as T,
          status: response.status,
          statusText: response.statusText,
          headers: this.parseHeaders(response.headers)
        };
      }
    }

    // Handle error responses
    try {
      const errorData = await response.json();
      throw createExceptionFromResponse(response.status, errorData, requestId);
    } catch (error) {
      // If error is already a KnowledgeBaseError, re-throw it
      if (error instanceof KnowledgeBaseError) {
        throw error;
      }

      // Otherwise create a generic API error
      const text = await response.text();
      throw createExceptionFromResponse(response.status, { message: text }, requestId);
    }
  }

  /**
   * Parse Headers object into a plain object.
   */
  private parseHeaders(headers: Headers): Record<string, string> {
    const result: Record<string, string> = {};
    headers.forEach((value, key) => {
      result[key] = value;
    });
    return result;
  }

  /**
   * Sleep for a specified number of milliseconds.
   */
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// ============================================================================
// Default HTTP Client Instance
// ============================================================================

let defaultHTTPClient: HTTPClient | null = null;

/**
 * Get or create the default HTTP client instance.
 */
export function getHTTPClient(config?: HTTPClientConfig): HTTPClient {
  if (!defaultHTTPClient || config) {
    defaultHTTPClient = new HTTPClient(config);
  }
  return defaultHTTPClient;
}

/**
 * Reset the default HTTP client instance.
 */
export function resetHTTPClient(): void {
  defaultHTTPClient = null;
}