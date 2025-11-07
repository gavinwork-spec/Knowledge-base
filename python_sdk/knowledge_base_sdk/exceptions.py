"""
Custom exceptions for the Knowledge Base API SDK.

Provides specific exception types for different error scenarios that can occur
when interacting with the API, making error handling more precise and informative.
"""

from typing import Any, Dict, Optional


class KnowledgeBaseError(Exception):
    """Base exception for all Knowledge Base API errors"""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_data: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response_data = response_data or {}
        self.request_id = request_id

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"status_code={self.status_code}, "
            f"request_id={self.request_id!r}"
            f")"
        )


class AuthenticationError(KnowledgeBaseError):
    """Raised when authentication fails (401, 403)"""

    def __init__(
        self,
        message: str = "Authentication failed",
        status_code: int = 401,
        response_data: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        super().__init__(message, status_code, response_data, request_id)


class APIError(KnowledgeBaseError):
    """Raised for general API errors (4xx, 5xx)"""

    def __init__(
        self,
        message: str,
        status_code: int,
        response_data: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        super().__init__(message, status_code, response_data, request_id)


class RateLimitError(KnowledgeBaseError):
    """Raised when rate limit is exceeded (429)"""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None,
        response_data: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        super().__init__(message, 429, response_data, request_id)
        self.retry_after = retry_after


class ValidationError(KnowledgeBaseError):
    """Raised when request validation fails (400)"""

    def __init__(
        self,
        message: str = "Validation failed",
        validation_errors: Optional[Dict[str, Any]] = None,
        response_data: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        super().__init__(message, 400, response_data, request_id)
        self.validation_errors = validation_errors or {}


class NotFoundError(KnowledgeBaseError):
    """Raised when a resource is not found (404)"""

    def __init__(
        self,
        message: str = "Resource not found",
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        response_data: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        super().__init__(message, 404, response_data, request_id)
        self.resource_type = resource_type
        self.resource_id = resource_id


class ConnectionError(KnowledgeBaseError):
    """Raised when connection to the API fails"""

    def __init__(
        self,
        message: str = "Failed to connect to API",
        original_exception: Optional[Exception] = None
    ):
        super().__init__(message)
        self.original_exception = original_exception


class TimeoutError(KnowledgeBaseError):
    """Raised when a request times out"""

    def __init__(
        self,
        message: str = "Request timed out",
        timeout_seconds: Optional[float] = None
    ):
        super().__init__(message)
        self.timeout_seconds = timeout_seconds


class WebSocketError(KnowledgeBaseError):
    """Raised for WebSocket-related errors"""

    def __init__(
        self,
        message: str = "WebSocket error occurred",
        code: Optional[int] = None,
        reason: Optional[str] = None
    ):
        super().__init__(message)
        self.code = code
        self.reason = reason


class ConsentRequiredError(AuthenticationError):
    """Raised when user consent is required for personalized search"""

    def __init__(
        self,
        message: str = "User consent required for personalized search",
        user_id: Optional[str] = None,
        response_data: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        super().__init__(message, 403, response_data, request_id)
        self.user_id = user_id


class PersonalizationDisabledError(AuthenticationError):
    """Raised when personalization is disabled for a user"""

    def __init__(
        self,
        message: str = "Personalization is disabled for this user",
        user_id: Optional[str] = None,
        response_data: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        super().__init__(message, 403, response_data, request_id)
        self.user_id = user_id


class ServiceUnavailableError(KnowledgeBaseError):
    """Raised when the service is temporarily unavailable (503)"""

    def __init__(
        self,
        message: str = "Service temporarily unavailable",
        retry_after: Optional[int] = None,
        response_data: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        super().__init__(message, 503, response_data, request_id)
        self.retry_after = retry_after


class DocumentIndexError(KnowledgeBaseError):
    """Raised when document indexing fails"""

    def __init__(
        self,
        message: str = "Document indexing failed",
        document_id: Optional[str] = None,
        response_data: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        super().__init__(message, response_data=response_data, request_id=request_id)
        self.document_id = document_id


class SearchError(KnowledgeBaseError):
    """Raised when search operation fails"""

    def __init__(
        self,
        message: str = "Search operation failed",
        query: Optional[str] = None,
        search_strategy: Optional[str] = None,
        response_data: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        super().__init__(message, response_data=response_data, request_id=request_id)
        self.query = query
        self.search_strategy = search_strategy


# Exception mapping for HTTP status codes
HTTP_STATUS_EXCEPTIONS = {
    400: ValidationError,
    401: AuthenticationError,
    403: AuthenticationError,
    404: NotFoundError,
    429: RateLimitError,
    500: APIError,
    502: ServiceUnavailableError,
    503: ServiceUnavailableError,
    504: TimeoutError,
}


def create_exception_from_response(
    status_code: int,
    response_data: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None
) -> KnowledgeBaseError:
    """
    Create appropriate exception from HTTP response.

    Args:
        status_code: HTTP status code
        response_data: Response JSON data
        request_id: Request ID for tracking

    Returns:
        Appropriate KnowledgeBaseError subclass
    """
    response_data = response_data or {}
    message = response_data.get("message", f"HTTP {status_code} error")

    # Get appropriate exception class
    exception_class = HTTP_STATUS_EXCEPTIONS.get(status_code, APIError)

    # Handle specific cases with additional context
    if status_code == 403:
        if "consent" in message.lower():
            exception_class = ConsentRequiredError
        elif "personalization" in message.lower():
            exception_class = PersonalizationDisabledError

    if status_code == 429:
        retry_after = response_data.get("retry_after")
        return RateLimitError(
            message=message,
            retry_after=retry_after,
            response_data=response_data,
            request_id=request_id
        )

    if status_code == 503:
        retry_after = response_data.get("retry_after")
        return ServiceUnavailableError(
            message=message,
            retry_after=retry_after,
            response_data=response_data,
            request_id=request_id
        )

    # Create exception with appropriate parameters
    if exception_class == ValidationError:
        validation_errors = response_data.get("errors", {})
        return ValidationError(
            message=message,
            validation_errors=validation_errors,
            response_data=response_data,
            request_id=request_id
        )

    if exception_class == NotFoundError:
        resource_type = response_data.get("resource_type")
        resource_id = response_data.get("resource_id")
        return NotFoundError(
            message=message,
            resource_type=resource_type,
            resource_id=resource_id,
            response_data=response_data,
            request_id=request_id
        )

    return exception_class(
        message=message,
        status_code=status_code,
        response_data=response_data,
        request_id=request_id
    )