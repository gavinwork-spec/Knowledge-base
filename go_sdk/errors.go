package kbsdk

import (
	"fmt"
)

// ============================================================================
// Base Error Types
// ============================================================================

// Error represents a Knowledge Base API error
type Error struct {
	Message    string `json:"message"`
	Status     *int   `json:"status,omitempty"`
	Code       *string `json:"code,omitempty"`
	Response   string `json:"response,omitempty"`
	RequestID  *string `json:"requestId,omitempty"`
	StackTrace string `json:"stackTrace,omitempty"`
}

// Error implements the error interface
func (e *Error) Error() string {
	if e.Status != nil {
		return fmt.Sprintf("API Error %d: %s", *e.Status, e.Message)
	}
	return fmt.Sprintf("API Error: %s", e.Message)
}

// Unwrap returns the underlying error message
func (e *Error) Unwrap() string {
	return e.Message
}

// ============================================================================
// Specific Error Types
// ============================================================================

// AuthenticationError represents authentication failures
type AuthenticationError struct {
	*Error
	UserID *string `json:"userId,omitempty"`
}

func (e *AuthenticationError) Error() string {
	if e.UserID != nil {
		return fmt.Sprintf("Authentication failed for user %s: %s", *e.UserID, e.Message)
	}
	return fmt.Sprintf("Authentication failed: %s", e.Message)
}

// ConsentRequiredError represents consent requirement for personalized search
type ConsentRequiredError struct {
	*Error
	UserID *string `json:"userId,omitempty"`
}

func (e *ConsentRequiredError) Error() string {
	if e.UserID != nil {
		return fmt.Sprintf("User consent required for user %s: %s", *e.UserID, e.Message)
	}
	return fmt.Sprintf("User consent required: %s", e.Message)
}

// PersonalizationDisabledError represents disabled personalization
type PersonalizationDisabledError struct {
	*Error
	UserID *string `json:"userId,omitempty"`
}

func (e *PersonalizationDisabledError) Error() string {
	if e.UserID != nil {
		return fmt.Sprintf("Personalization disabled for user %s: %s", *e.UserID, e.Message)
	}
	return fmt.Sprintf("Personalization disabled: %s", e.Message)
}

// ValidationError represents validation failures
type ValidationError struct {
	*Error
	ValidationErrors map[string]interface{} `json:"validationErrors,omitempty"`
}

func (e *ValidationError) Error() string {
	return fmt.Sprintf("Validation failed: %s", e.Message)
}

// NotFoundError represents resource not found errors
type NotFoundError struct {
	*Error
	ResourceType *string `json:"resourceType,omitempty"`
	ResourceID   *string `json:"resourceId,omitempty"`
}

func (e *NotFoundError) Error() string {
	if e.ResourceType != nil && e.ResourceID != nil {
		return fmt.Sprintf("Resource %s with ID %s not found: %s", *e.ResourceType, *e.ResourceID, e.Message)
	} else if e.ResourceType != nil {
		return fmt.Sprintf("Resource %s not found: %s", *e.ResourceType, e.Message)
	}
	return fmt.Sprintf("Resource not found: %s", e.Message)
}

// RateLimitError represents rate limit exceeded errors
type RateLimitError struct {
	*Error
	RetryAfter *int `json:"retryAfter,omitempty"`
}

func (e *RateLimitError) Error() string {
	if e.RetryAfter != nil {
		return fmt.Sprintf("Rate limit exceeded. Retry after %d seconds: %s", *e.RetryAfter, e.Message)
	}
	return fmt.Sprintf("Rate limit exceeded: %s", e.Message)
}

// ServiceUnavailableError represents service unavailable errors
type ServiceUnavailableError struct {
	*Error
	RetryAfter *int `json:"retryAfter,omitempty"`
}

func (e *ServiceUnavailableError) Error() string {
	if e.RetryAfter != nil {
		return fmt.Sprintf("Service temporarily unavailable. Retry after %d seconds: %s", *e.RetryAfter, e.Message)
	}
	return fmt.Sprintf("Service temporarily unavailable: %s", e.Message)
}

// ConnectionError represents connection failures
type ConnectionError struct {
	*Error
	OriginalError error `json:"originalError,omitempty"`
}

func (e *ConnectionError) Error() string {
	if e.OriginalError != nil {
		return fmt.Sprintf("Connection failed: %s (caused by: %v)", e.Message, e.OriginalError)
	}
	return fmt.Sprintf("Connection failed: %s", e.Message)
}

// TimeoutError represents timeout errors
type TimeoutError struct {
	*Error
	TimeoutSeconds *float64 `json:"timeoutSeconds,omitempty"`
}

func (e *TimeoutError) Error() string {
	if e.TimeoutSeconds != nil {
		return fmt.Sprintf("Request timed out after %.2f seconds: %s", *e.TimeoutSeconds, e.Message)
	}
	return fmt.Sprintf("Request timed out: %s", e.Message)
}

// WebSocketError represents WebSocket errors
type WebSocketError struct {
	*Error
	Code   *int    `json:"code,omitempty"`
	Reason *string `json:"reason,omitempty"`
}

func (e *WebSocketError) Error() string {
	if e.Code != nil {
		return fmt.Sprintf("WebSocket error (code %d): %s", *e.Code, e.Message)
	}
	return fmt.Sprintf("WebSocket error: %s", e.Message)
}

// DocumentIndexError represents document indexing failures
type DocumentIndexError struct {
	*Error
	DocumentID *string `json:"documentId,omitempty"`
}

func (e *DocumentIndexError) Error() string {
	if e.DocumentID != nil {
		return fmt.Sprintf("Document indexing failed for %s: %s", *e.DocumentID, e.Message)
	}
	return fmt.Sprintf("Document indexing failed: %s", e.Message)
}

// SearchError represents search operation failures
type SearchError struct {
	*Error
	Query          *string `json:"query,omitempty"`
	SearchStrategy *string `json:"searchStrategy,omitempty"`
}

func (e *SearchError) Error() string {
	if e.Query != nil && e.SearchStrategy != nil {
		return fmt.Sprintf("Search failed for query '%s' with strategy %s: %s", *e.Query, *e.SearchStrategy, e.Message)
	} else if e.Query != nil {
		return fmt.Sprintf("Search failed for query '%s': %s", *e.Query, e.Message)
	}
	return fmt.Sprintf("Search failed: %s", e.Message)
}

// ============================================================================
// Error Factory Functions
// ============================================================================

// HTTP status code to error type mapping
var httpStatusExceptions = map[int]func(string, *int, string, *string) error{
	400: newValidationError,
	401: newAuthenticationError,
	403: newAuthenticationError,
	404: newNotFoundError,
	429: newRateLimitError,
	500: newAPIError,
	502: newServiceUnavailableError,
	503: newServiceUnavailableError,
	504: newTimeoutError,
}

// createExceptionFromResponse creates appropriate error from HTTP response
func createExceptionFromResponse(status int, response string, requestID *string) error {
	message := response
	if message == "" {
		message = fmt.Sprintf("HTTP %d error", status)
	}

	// Get appropriate error constructor
	constructor, exists := httpStatusExceptions[status]
	if !exists {
		constructor = newAPIError
	}

	// Handle specific cases with additional context
	if status == 403 {
		if containsIgnoreCase(message, "consent") {
			return newConsentRequiredError(message, nil, response, requestID)
		}
		if containsIgnoreCase(message, "personalization") {
			return newPersonalizationDisabledError(message, nil, response, requestID)
		}
	}

	return constructor(message, &status, response, requestID)
}

// ============================================================================
// Error Constructor Functions
// ============================================================================

func newAPIError(message string, status *int, response string, requestID *string) error {
	return &Error{
		Message:   message,
		Status:    status,
		Response:  response,
		RequestID: requestID,
	}
}

func newValidationError(message string, status *int, response string, requestID *string) error {
	return &ValidationError{
		Error: &Error{
			Message:   message,
			Status:    status,
			Response:  response,
			RequestID: requestID,
		},
	}
}

func newAuthenticationError(message string, status *int, response string, requestID *string) error {
	return &AuthenticationError{
		Error: &Error{
			Message:   message,
			Status:    status,
			Response:  response,
			RequestID: requestID,
		},
	}
}

func newConsentRequiredError(message string, status *int, response string, requestID *string) error {
	return &ConsentRequiredError{
		Error: &Error{
			Message:   message,
			Status:    status,
			Response:  response,
			RequestID: requestID,
		},
	}
}

func newPersonalizationDisabledError(message string, status *int, response string, requestID *string) error {
	return &PersonalizationDisabledError{
		Error: &Error{
			Message:   message,
			Status:    status,
			Response:  response,
			RequestID: requestID,
		},
	}
}

func newNotFoundError(message string, status *int, response string, requestID *string) error {
	return &NotFoundError{
		Error: &Error{
			Message:   message,
			Status:    status,
			Response:  response,
			RequestID: requestID,
		},
	}
}

func newRateLimitError(message string, status *int, response string, requestID *string) error {
	return &RateLimitError{
		Error: &Error{
			Message:   message,
			Status:    status,
			Response:  response,
			RequestID: requestID,
		},
	}
}

func newServiceUnavailableError(message string, status *int, response string, requestID *string) error {
	return &ServiceUnavailableError{
		Error: &Error{
			Message:   message,
			Status:    status,
			Response:  response,
			RequestID: requestID,
		},
	}
}

func newTimeoutError(message string, status *int, response string, requestID *string) error {
	return &TimeoutError{
		Error: &Error{
			Message:   message,
			Status:    status,
			Response:  response,
			RequestID: requestID,
		},
	}
}

func newConnectionError(message string, originalErr error) error {
	return &ConnectionError{
		Error: &Error{
			Message: message,
		},
		OriginalError: originalErr,
	}
}

func newWebSocketError(message string, code *int, reason *string) error {
	return &WebSocketError{
		Error: &Error{
			Message: message,
		},
		Code:   code,
		Reason: reason,
	}
}

func newDocumentIndexError(message string, documentID *string) error {
	return &DocumentIndexError{
		Error: &Error{
			Message: message,
		},
		DocumentID: documentID,
	}
}

func newSearchError(message string, query *string, strategy *string) error {
	return &SearchError{
		Error: &Error{
			Message: message,
		},
		Query:          query,
		SearchStrategy: strategy,
	}
}

// ============================================================================
// Type Checking Functions
// ============================================================================

// IsAPIError checks if error is an API error
func IsAPIError(err error) bool {
	_, ok := err.(*Error)
	return ok
}

// IsValidationError checks if error is a validation error
func IsValidationError(err error) bool {
	_, ok := err.(*ValidationError)
	return ok
}

// IsAuthenticationError checks if error is an authentication error
func IsAuthenticationError(err error) bool {
	_, ok := err.(*AuthenticationError)
	return ok
}

// IsConsentRequiredError checks if error is a consent required error
func IsConsentRequiredError(err error) bool {
	_, ok := err.(*ConsentRequiredError)
	return ok
}

// IsPersonalizationDisabledError checks if error is a personalization disabled error
func IsPersonalizationDisabledError(err error) bool {
	_, ok := err.(*PersonalizationDisabledError)
	return ok
}

// IsNotFoundError checks if error is a not found error
func IsNotFoundError(err error) bool {
	_, ok := err.(*NotFoundError)
	return ok
}

// IsRateLimitError checks if error is a rate limit error
func IsRateLimitError(err error) bool {
	_, ok := err.(*RateLimitError)
	return ok
}

// IsServiceUnavailableError checks if error is a service unavailable error
func IsServiceUnavailableError(err error) bool {
	_, ok := err.(*ServiceUnavailableError)
	return ok
}

// IsConnectionError checks if error is a connection error
func IsConnectionError(err error) bool {
	_, ok := err.(*ConnectionError)
	return ok
}

// IsTimeoutError checks if error is a timeout error
func IsTimeoutError(err error) bool {
	_, ok := err.(*TimeoutError)
	return ok
}

// IsWebSocketError checks if error is a WebSocket error
func IsWebSocketError(err error) bool {
	_, ok := err.(*WebSocketError)
	return ok
}

// IsDocumentIndexError checks if error is a document index error
func IsDocumentIndexError(err error) bool {
	_, ok := err.(*DocumentIndexError)
	return ok
}

// IsSearchError checks if error is a search error
func IsSearchError(err error) bool {
	_, ok := err.(*SearchError)
	return ok
}

// ============================================================================
// Utility Functions
// ============================================================================

// containsIgnoreCase checks if string contains substring (case insensitive)
func containsIgnoreCase(s, substr string) bool {
	return len(s) >= len(substr) &&
		   (s == substr ||
		    len(s) > len(substr) &&
		    (s[:len(substr)] == substr ||
		     s[len(s)-len(substr):] == substr ||
		     containsIgnoreCaseInternal(s, substr)))
}

func containsIgnoreCaseInternal(s, substr string) bool {
	s = toLower(s)
	substr = toLower(substr)
	return len(s) >= len(substr) && findSubstring(s, substr) >= 0
}

func toLower(s string) string {
	result := make([]byte, len(s))
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c >= 'A' && c <= 'Z' {
			result[i] = c + 32
		} else {
			result[i] = c
		}
	}
	return string(result)
}

func findSubstring(s, substr string) int {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return i
		}
	}
	return -1
}