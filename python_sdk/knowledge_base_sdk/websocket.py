"""
WebSocket client for real-time interactions with the Knowledge Base API.

Provides asynchronous WebSocket connections for real-time search, suggestions,
and live updates with automatic reconnection and error handling.
"""

import asyncio
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Union
from datetime import datetime

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from .exceptions import WebSocketError
from .types import (
    WebSocketMessage,
    WebSocketSearchMessage,
    WebSocketSearchResultsMessage,
    WebSocketProgressMessage,
    WebSocketSuggestionsMessage,
    WebSocketErrorMessage,
    SearchResult,
    SearchAnalytics,
)

logger = logging.getLogger(__name__)


class WebSocketClient:
    """
    WebSocket client for real-time API interactions.

    Supports both regular search and personalized search WebSocket connections
    with automatic reconnection, message validation, and event handling.
    """

    def __init__(
        self,
        base_url: str = "ws://localhost:8006",
        personalized_url: str = "ws://localhost:8007",
        api_key: Optional[str] = None,
        ping_interval: int = 20,
        ping_timeout: int = 20,
        close_timeout: int = 10,
        max_reconnect_attempts: int = 5,
        reconnect_delay: float = 1.0,
    ):
        """
        Initialize WebSocket client.

        Args:
            base_url: Base WebSocket URL for search
            personalized_url: WebSocket URL for personalized search
            api_key: API key for authentication
            ping_interval: Interval in seconds for ping frames
            ping_timeout: Timeout in seconds for ping response
            close_timeout: Timeout in seconds for close handshake
            max_reconnect_attempts: Maximum reconnection attempts
            reconnect_delay: Delay between reconnection attempts
        """
        self.base_url = base_url.rstrip("/")
        self.personalized_url = personalized_url.rstrip("/")
        self.api_key = api_key
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self.close_timeout = close_timeout
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay

        # Connection state
        self._search_connection: Optional[websockets.WebSocketClientProtocol] = None
        self._personalized_connection: Optional[websockets.WebSocketClientProtocol] = None
        self._connected = {"search": False, "personalized": False}
        self._reconnecting = {"search": False, "personalized": False}

        # Event handlers
        self._handlers: Dict[str, List[Callable]] = {
            "open": [],
            "close": [],
            "error": [],
            "message": [],
            "results": [],
            "progress": [],
            "suggestions": [],
            "search_error": [],
        }

        # Background tasks
        self._listener_tasks: Dict[str, asyncio.Task] = {}
        self._running = False

    def on(self, event: str, handler: Callable):
        """
        Register event handler.

        Args:
            event: Event name (open, close, error, message, results, progress, suggestions, search_error)
            handler: Handler function
        """
        if event not in self._handlers:
            raise ValueError(f"Unknown event: {event}")

        self._handlers[event].append(handler)

    def off(self, event: str, handler: Callable):
        """
        Remove event handler.

        Args:
            event: Event name
            handler: Handler function to remove
        """
        if event in self._handlers:
            try:
                self._handlers[event].remove(handler)
            except ValueError:
                pass

    def _emit(self, event: str, *args, **kwargs):
        """Emit event to all registered handlers"""
        for handler in self._handlers[event]:
            try:
                if asyncio.iscoroutinefunction(handler):
                    asyncio.create_task(handler(*args, **kwargs))
                else:
                    handler(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in event handler for {event}: {e}")

    async def connect_search(self) -> None:
        """Connect to search WebSocket"""
        if self._connected["search"] or self._reconnecting["search"]:
            return

        self._reconnecting["search"] = True
        attempt = 0

        while attempt < self.max_reconnect_attempts:
            try:
                logger.info(f"Connecting to search WebSocket at {self.base_url}/ws/search")
                self._search_connection = await websockets.connect(
                    f"{self.base_url}/ws/search",
                    ping_interval=self.ping_interval,
                    ping_timeout=self.ping_timeout,
                    close_timeout=self.close_timeout,
                    extra_headers=self._build_headers(),
                )

                self._connected["search"] = True
                self._reconnecting["search"] = False
                self._emit("open", "search")

                # Start listener task
                if "search" not in self._listener_tasks or self._listener_tasks["search"].done():
                    self._listener_tasks["search"] = asyncio.create_task(
                        self._listen_messages("search")
                    )

                logger.info("Connected to search WebSocket")
                return

            except Exception as e:
                attempt += 1
                logger.warning(
                    f"Failed to connect to search WebSocket (attempt {attempt}/{self.max_reconnect_attempts}): {e}"
                )
                if attempt < self.max_reconnect_attempts:
                    await asyncio.sleep(self.reconnect_delay * (2 ** (attempt - 1)))  # Exponential backoff

        self._reconnecting["search"] = False
        error = WebSocketError(f"Failed to connect to search WebSocket after {attempt} attempts")
        self._emit("error", error)
        raise error

    async def connect_personalized(self) -> None:
        """Connect to personalized search WebSocket"""
        if self._connected["personalized"] or self._reconnecting["personalized"]:
            return

        self._reconnecting["personalized"] = True
        attempt = 0

        while attempt < self.max_reconnect_attempts:
            try:
                logger.info(f"Connecting to personalized WebSocket at {self.personalized_url}/ws/personalized-search")
                self._personalized_connection = await websockets.connect(
                    f"{self.personalized_url}/ws/personalized-search",
                    ping_interval=self.ping_interval,
                    ping_timeout=self.ping_timeout,
                    close_timeout=self.close_timeout,
                    extra_headers=self._build_headers(),
                )

                self._connected["personalized"] = True
                self._reconnecting["personalized"] = False
                self._emit("open", "personalized")

                # Start listener task
                if "personalized" not in self._listener_tasks or self._listener_tasks["personalized"].done():
                    self._listener_tasks["personalized"] = asyncio.create_task(
                        self._listen_messages("personalized")
                    )

                logger.info("Connected to personalized search WebSocket")
                return

            except Exception as e:
                attempt += 1
                logger.warning(
                    f"Failed to connect to personalized WebSocket (attempt {attempt}/{self.max_reconnect_attempts}): {e}"
                )
                if attempt < self.max_reconnect_attempts:
                    await asyncio.sleep(self.reconnect_delay * (2 ** (attempt - 1)))  # Exponential backoff

        self._reconnecting["personalized"] = False
        error = WebSocketError(f"Failed to connect to personalized WebSocket after {attempt} attempts")
        self._emit("error", error)
        raise error

    def _build_headers(self) -> Dict[str, str]:
        """Build WebSocket headers including authentication"""
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _listen_messages(self, connection_type: str):
        """Listen for messages on a WebSocket connection"""
        connection = (
            self._search_connection if connection_type == "search"
            else self._personalized_connection
        )

        try:
            async for message in connection:
                try:
                    data = json.loads(message)
                    await self._handle_message(data, connection_type)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON message: {e}")
                    error = WebSocketError(f"Invalid JSON message: {message}")
                    self._emit("search_error", error)
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
                    error = WebSocketError(f"Message handling error: {str(e)}")
                    self._emit("search_error", error)

        except ConnectionClosed:
            logger.info(f"{connection_type.title()} WebSocket connection closed")
            self._connected[connection_type] = False
            self._emit("close", connection_type)

            # Attempt reconnection
            if self._running:
                await self._reconnect(connection_type)

        except Exception as e:
            logger.error(f"WebSocket error for {connection_type}: {e}")
            self._connected[connection_type] = False
            error = WebSocketError(f"WebSocket error: {str(e)}")
            self._emit("error", error)

    async def _handle_message(self, data: Dict[str, Any], connection_type: str):
        """Handle incoming WebSocket message"""
        message_type = data.get("type")
        if not message_type:
            logger.warning("Received message without type")
            return

        self._emit("message", data, connection_type)

        # Handle specific message types
        if message_type == "results":
            await self._handle_search_results(data)
        elif message_type == "progress":
            await self._handle_progress_message(data)
        elif message_type == "suggestions":
            await self._handle_suggestions(data)
        elif message_type == "error":
            await self._handle_error_message(data)

    async def _handle_search_results(self, data: Dict[str, Any]):
        """Handle search results message"""
        try:
            # Parse search results
            results_data = data.get("results", [])
            results = []
            for result_data in results_data:
                # Create SearchResult objects
                result = SearchResult(
                    id=result_data.get("id", ""),
                    title=result_data.get("title", ""),
                    content=result_data.get("content", ""),
                    score=result_data.get("score", 0.0),
                    metadata=result_data.get("metadata", {}),
                    source_type=result_data.get("source_type", "semantic"),
                    explanation=result_data.get("explanation"),
                )
                results.append(result)

            # Parse analytics if present
            analytics = None
            analytics_data = data.get("analytics")
            if analytics_data:
                analytics = SearchAnalytics(**analytics_data)

            # Emit results event
            self._emit("results", {
                "search_id": data.get("search_id"),
                "results": results,
                "execution_time": data.get("execution_time", 0.0),
                "analytics": analytics,
                "timestamp": datetime.now(),
            })

        except Exception as e:
            logger.error(f"Error handling search results: {e}")
            error = WebSocketError(f"Search results parsing error: {str(e)}")
            self._emit("search_error", error)

    async def _handle_progress_message(self, data: Dict[str, Any]):
        """Handle progress message"""
        self._emit("progress", {
            "search_id": data.get("search_id"),
            "status": data.get("status"),
            "message": data.get("message"),
            "timestamp": datetime.now(),
        })

    async def _handle_suggestions(self, data: Dict[str, Any]):
        """Handle suggestions message"""
        self._emit("suggestions", {
            "query": data.get("query"),
            "suggestions": data.get("suggestions", []),
            "personalized": data.get("personalized", False),
            "timestamp": datetime.now(),
        })

    async def _handle_error_message(self, data: Dict[str, Any]):
        """Handle error message"""
        error = WebSocketError(
            message=data.get("message", "Unknown WebSocket error"),
            code=data.get("code"),
            reason=data.get("reason")
        )
        self._emit("search_error", error)

    async def _reconnect(self, connection_type: str):
        """Attempt to reconnect a WebSocket connection"""
        if self._reconnecting[connection_type]:
            return

        try:
            if connection_type == "search":
                await self.connect_search()
            else:
                await self.connect_personalized()
        except Exception as e:
            logger.error(f"Failed to reconnect {connection_type} WebSocket: {e}")

    async def send_search_request(
        self,
        query: str,
        strategy: str = "unified",
        top_k: int = 10,
        threshold: float = 0.7,
        connection_type: str = "search"
    ) -> None:
        """
        Send search request via WebSocket.

        Args:
            query: Search query
            strategy: Search strategy
            top_k: Number of results
            threshold: Similarity threshold
            connection_type: Connection type ("search" or "personalized")
        """
        if not self._connected[connection_type]:
            if connection_type == "search":
                await self.connect_search()
            else:
                await self.connect_personalized()

        message = WebSocketSearchMessage(
            type="search",
            query=query,
            strategy=strategy,
            top_k=top_k,
            threshold=threshold
        )

        connection = (
            self._search_connection if connection_type == "search"
            else self._personalized_connection
        )

        try:
            await connection.send(message.json())
        except Exception as e:
            logger.error(f"Failed to send search request: {e}")
            raise WebSocketError(f"Failed to send search request: {str(e)}")

    async def send_suggestion_request(
        self,
        query: str,
        user_id: Optional[str] = None,
        max_suggestions: int = 5
    ) -> None:
        """
        Send suggestion request via WebSocket.

        Args:
            query: Partial query for suggestions
            user_id: User ID for personalized suggestions
            max_suggestions: Maximum number of suggestions
        """
        if not self._connected["personalized"]:
            await self.connect_personalized()

        message = {
            "type": "suggestions",
            "query": query,
            "user_id": user_id,
            "max_suggestions": max_suggestions,
            "timestamp": datetime.now().isoformat(),
        }

        try:
            await self._personalized_connection.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send suggestion request: {e}")
            raise WebSocketError(f"Failed to send suggestion request: {str(e)}")

    async def start(self) -> None:
        """Start WebSocket client and connect to both endpoints"""
        self._running = True
        logger.info("Starting WebSocket client")

        # Connect to both endpoints
        await asyncio.gather(
            self.connect_search(),
            self.connect_personalized(),
            return_exceptions=True
        )

    async def stop(self) -> None:
        """Stop WebSocket client and close connections"""
        self._running = False
        logger.info("Stopping WebSocket client")

        # Cancel listener tasks
        for task in self._listener_tasks.values():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Close connections
        if self._search_connection:
            await self._search_connection.close()
        if self._personalized_connection:
            await self._personalized_connection.close()

        self._connected = {"search": False, "personalized": False}

    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop()


# Convenience functions for creating WebSocket clients
def create_search_websocket(
    base_url: str = "ws://localhost:8006",
    api_key: Optional[str] = None,
    **kwargs
) -> WebSocketClient:
    """
    Create a WebSocket client for search operations.

    Args:
        base_url: WebSocket base URL
        api_key: API key for authentication
        **kwargs: Additional WebSocket client options

    Returns:
        WebSocketClient instance
    """
    return WebSocketClient(base_url=base_url, api_key=api_key, **kwargs)


def create_personalized_websocket(
    base_url: str = "ws://localhost:8006",
    personalized_url: str = "ws://localhost:8007",
    api_key: Optional[str] = None,
    **kwargs
) -> WebSocketClient:
    """
    Create a WebSocket client for personalized search operations.

    Args:
        base_url: WebSocket base URL for search
        personalized_url: WebSocket URL for personalized search
        api_key: API key for authentication
        **kwargs: Additional WebSocket client options

    Returns:
        WebSocketClient instance
    """
    return WebSocketClient(
        base_url=base_url,
        personalized_url=personalized_url,
        api_key=api_key,
        **kwargs
    )