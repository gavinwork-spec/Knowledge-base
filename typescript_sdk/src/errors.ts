/**
 * Custom error classes for the Knowledge Base API TypeScript/JavaScript SDK.
 *
 * Provides specific error types for different error scenarios that can occur
 * when interacting with the API, making error handling more precise and informative.
 */

// ============================================================================
// Base Error Class
// ============================================================================

export class KnowledgeBaseError extends Error {
  public readonly status?: number;
  public readonly code?: string;
  public readonly response?: unknown;
  public readonly requestId?: string;

  constructor(
    message: string,
    options: {
      status?: number;
      code?: string;
      response?: unknown;
      requestId?: string;
    } = {}
  ) {
    super(message);
    this.name = 'KnowledgeBaseError';
    this.status = options.status;
    this.code = options.code;
    this.response = options.response;
    this.requestId = options.requestId;

    // Maintains proper stack trace for where our error was thrown (only available on V8)
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, KnowledgeBaseError);
    }
  }

  public toJSON(): Record<string, unknown> {
    return {
      name: this.name,
      message: this.message,
      status: this.status,
      code: this.code,
      response: this.response,
      requestId: this.requestId,
      stack: this.stack
    };
  }
}

// ============================================================================
// Authentication Errors
// ============================================================================

export class AuthenticationError extends KnowledgeBaseError {
  public readonly userId?: string;

  constructor(
    message: string = 'Authentication failed',
    options: {
      status?: number;
      userId?: string;
      response?: unknown;
      requestId?: string;
    } = {}
  ) {
    super(message, { status: 401, ...options });
    this.name = 'AuthenticationError';
    this.userId = options.userId;
  }
}

export class ConsentRequiredError extends AuthenticationError {
  constructor(
    message: string = 'User consent required for personalized search',
    options: {
      userId?: string;
      response?: unknown;
      requestId?: string;
    } = {}
  ) {
    super(message, { status: 403, ...options });
    this.name = 'ConsentRequiredError';
  }
}

export class PersonalizationDisabledError extends AuthenticationError {
  constructor(
    message: string = 'Personalization is disabled for this user',
    options: {
      userId?: string;
      response?: unknown;
      requestId?: string;
    } = {}
  ) {
    super(message, { status: 403, ...options });
    this.name = 'PersonalizationDisabledError';
  }
}

// ============================================================================
// HTTP Errors
// ============================================================================

export class APIError extends KnowledgeBaseError {
  constructor(
    message: string,
    status: number,
    options: {
      response?: unknown;
      requestId?: string;
    } = {}
  ) {
    super(message, { status, ...options });
    this.name = 'APIError';
  }
}

export class ValidationError extends KnowledgeBaseError {
  public readonly validationErrors?: Record<string, unknown>;

  constructor(
    message: string = 'Validation failed',
    options: {
      validationErrors?: Record<string, unknown>;
      response?: unknown;
      requestId?: string;
    } = {}
  ) {
    super(message, { status: 400, ...options });
    this.name = 'ValidationError';
    this.validationErrors = options.validationErrors;
  }
}

export class NotFoundError extends KnowledgeBaseError {
  public readonly resourceType?: string;
  public readonly resourceId?: string;

  constructor(
    message: string = 'Resource not found',
    options: {
      resourceType?: string;
      resourceId?: string;
      response?: unknown;
      requestId?: string;
    } = {}
  ) {
    super(message, { status: 404, ...options });
    this.name = 'NotFoundError';
    this.resourceType = options.resourceType;
    this.resourceId = options.resourceId;
  }
}

export class RateLimitError extends KnowledgeBaseError {
  public readonly retryAfter?: number;

  constructor(
    message: string = 'Rate limit exceeded',
    options: {
      retryAfter?: number;
      response?: unknown;
      requestId?: string;
    } = {}
  ) {
    super(message, { status: 429, ...options });
    this.name = 'RateLimitError';
    this.retryAfter = options.retryAfter;
  }
}

export class ServiceUnavailableError extends KnowledgeBaseError {
  public readonly retryAfter?: number;

  constructor(
    message: string = 'Service temporarily unavailable',
    options: {
      retryAfter?: number;
      response?: unknown;
      requestId?: string;
    } = {}
  ) {
    super(message, { status: 503, ...options });
    this.name = 'ServiceUnavailableError';
    this.retryAfter = options.retryAfter;
  }
}

// ============================================================================
// Network Errors
// ============================================================================

export class ConnectionError extends KnowledgeBaseError {
  public readonly originalError?: Error;

  constructor(
    message: string = 'Failed to connect to API',
    options: {
      originalError?: Error;
    } = {}
  ) {
    super(message);
    this.name = 'ConnectionError';
    this.originalError = options.originalError;
  }
}

export class TimeoutError extends KnowledgeBaseError {
  public readonly timeoutSeconds?: number;

  constructor(
    message: string = 'Request timed out',
    options: {
      timeoutSeconds?: number;
    } = {}
  ) {
    super(message);
    this.name = 'TimeoutError';
    this.timeoutSeconds = options.timeoutSeconds;
  }
}

// ============================================================================
// WebSocket Errors
// ============================================================================

export class WebSocketError extends KnowledgeBaseError {
  public readonly code?: number;
  public readonly reason?: string;

  constructor(
    message: string = 'WebSocket error occurred',
    options: {
      code?: number;
      reason?: string;
    } = {}
  ) {
    super(message);
    this.name = 'WebSocketError';
    this.code = options.code;
    this.reason = options.reason;
  }
}

// ============================================================================
// Feature-Specific Errors
// ============================================================================

export class DocumentIndexError extends KnowledgeBaseError {
  public readonly documentId?: string;

  constructor(
    message: string = 'Document indexing failed',
    options: {
      documentId?: string;
      response?: unknown;
      requestId?: string;
    } = {}
  ) {
    super(message, options);
    this.name = 'DocumentIndexError';
    this.documentId = options.documentId;
  }
}

export class SearchError extends KnowledgeBaseError {
  public readonly query?: string;
  public readonly searchStrategy?: string;

  constructor(
    message: string = 'Search operation failed',
    options: {
      query?: string;
      searchStrategy?: string;
      response?: unknown;
      requestId?: string;
    } = {}
  ) {
    super(message, options);
    this.name = 'SearchError';
    this.query = options.query;
    this.searchStrategy = options.searchStrategy;
  }
}

// ============================================================================
// Error Factory Functions
// ============================================================================

const HTTP_STATUS_EXCEPTIONS: Record<number, typeof KnowledgeBaseError> = {
  400: ValidationError,
  401: AuthenticationError,
  403: AuthenticationError,
  404: NotFoundError,
  429: RateLimitError,
  500: APIError,
  502: ServiceUnavailableError,
  503: ServiceUnavailableError,
  504: TimeoutError,
};

/**
 * Create appropriate exception from HTTP response.
 */
export function createExceptionFromResponse(
  status: number,
  response?: unknown,
  requestId?: string
): KnowledgeBaseError {
  const responseData = response as Record<string, unknown> || {};
  const message = typeof responseData.message === 'string'
    ? responseData.message
    : `HTTP ${status} error`;

  // Get appropriate exception class
  let ExceptionClass = HTTP_STATUS_EXCEPTIONS[status] || APIError;

  // Handle specific cases with additional context
  if (status === 403) {
    if (typeof message === 'string' && message.toLowerCase().includes('consent')) {
      ExceptionClass = ConsentRequiredError;
    } else if (typeof message === 'string' && message.toLowerCase().includes('personalization')) {
      ExceptionClass = PersonalizationDisabledError;
    }
  }

  // Create exceptions with additional context
  switch (ExceptionClass) {
    case RateLimitError:
      return new RateLimitError(message, {
        retryAfter: typeof responseData.retry_after === 'number' ? responseData.retry_after : undefined,
        response: responseData,
        requestId
      });

    case ServiceUnavailableError:
      return new ServiceUnavailableError(message, {
        retryAfter: typeof responseData.retry_after === 'number' ? responseData.retry_after : undefined,
        response: responseData,
        requestId
      });

    case ValidationError:
      return new ValidationError(message, {
        validationErrors: typeof responseData.errors === 'object' ? responseData.errors as Record<string, unknown> : undefined,
        response: responseData,
        requestId
      });

    case NotFoundError:
      return new NotFoundError(message, {
        resourceType: typeof responseData.resource_type === 'string' ? responseData.resource_type : undefined,
        resourceId: typeof responseData.resource_id === 'string' ? responseData.resource_id : undefined,
        response: responseData,
        requestId
      });

    case ConsentRequiredError:
      return new ConsentRequiredError(message, {
        userId: typeof responseData.user_id === 'string' ? responseData.user_id : undefined,
        response: responseData,
        requestId
      });

    case PersonalizationDisabledError:
      return new PersonalizationDisabledError(message, {
        userId: typeof responseData.user_id === 'string' ? responseData.user_id : undefined,
        response: responseData,
        requestId
      });

    default:
      return new ExceptionClass(message, {
        status,
        response: responseData,
        requestId
      });
  }
}

/**
 * Check if an error is a specific type.
 */
export function isInstanceOf(error: unknown, ErrorClass: typeof KnowledgeBaseError): error is InstanceType<typeof ErrorClass> {
  return error instanceof ErrorClass;
}

/**
 * Type guard for API errors.
 */
export function isAPIError(error: unknown): error is KnowledgeBaseError {
  return error instanceof KnowledgeBaseError;
}

/**
 * Type guard for validation errors.
 */
export function isValidationError(error: unknown): error is ValidationError {
  return error instanceof ValidationError;
}

/**
 * Type guard for authentication errors.
 */
export function isAuthenticationError(error: unknown): error is AuthenticationError {
  return error instanceof AuthenticationError;
}

/**
 * Type guard for rate limit errors.
 */
export function isRateLimitError(error: unknown): error is RateLimitError {
  return error instanceof RateLimitError;
}

/**
 * Type guard for WebSocket errors.
 */
export function isWebSocketError(error: unknown): error is WebSocketError {
  return error instanceof WebSocketError;
}