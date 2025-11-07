"""
Search Performance Optimization Module

Provides advanced optimization techniques for the hybrid search engine
including caching, indexing optimizations, and performance monitoring.
"""

import asyncio
import time
import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, OrderedDict
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging

from hybrid_search_engine import SearchResult, SearchRequest


@dataclass
class CacheConfig:
    """Configuration for caching systems"""
    enabled: bool = True
    max_size: int = 1000
    ttl_seconds: int = 3600  # 1 hour
    cleanup_interval: int = 300  # 5 minutes
    memory_limit_mb: int = 512


@dataclass
class OptimizationConfig:
    """Configuration for search optimizations"""
    cache_config: CacheConfig = CacheConfig()
    enable_parallel_search: bool = True
    max_workers: int = 4
    enable_result_caching: bool = True
    enable_query_cache: bool = True
    enable_compression: bool = True
    batch_size: int = 100
    prefetch_enabled: bool = True
    prefetch_count: int = 5


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring"""
    total_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    avg_query_time: float = 0.0
    avg_indexing_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    concurrent_queries: int = 0
    error_count: int = 0
    last_updated: datetime = None

    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()


class LRUCache:
    """Thread-safe LRU cache implementation"""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = OrderedDict()
        self.timestamps = {}
        self.lock = threading.RLock()
        self._last_cleanup = time.time()

    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired"""
        if key not in self.timestamps:
            return True
        return time.time() - self.timestamps[key] > self.ttl_seconds

    def _cleanup_expired(self):
        """Remove expired entries"""
        current_time = time.time()
        if current_time - self._last_cleanup < 60:  # Cleanup every minute
            return

        with self.lock:
            expired_keys = [
                key for key in self.cache.keys()
                if self._is_expired(key)
            ]
            for key in expired_keys:
                self.cache.pop(key, None)
                self.timestamps.pop(key, None)
            self._last_cleanup = current_time

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            self._cleanup_expired()

            if key in self.cache and not self._is_expired(key):
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                return value
            return None

    def put(self, key: str, value: Any):
        """Put value in cache"""
        with self.lock:
            self._cleanup_expired()

            if key in self.cache:
                # Update existing entry
                self.cache.pop(key)
            elif len(self.cache) >= self.max_size:
                # Remove least recently used
                oldest_key = next(iter(self.cache))
                self.cache.pop(oldest_key)
                self.timestamps.pop(oldest_key, None)

            self.cache[key] = value
            self.timestamps[key] = time.time()

    def clear(self):
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()

    def size(self) -> int:
        """Get current cache size"""
        with self.lock:
            return len(self.cache)

    def hit_rate(self) -> float:
        """Calculate cache hit rate (placeholder)"""
        return 0.0  # Would need hit/miss counters


class QueryCache:
    """Specialized cache for search queries"""

    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache = LRUCache(config.max_size, config.ttl_seconds)
        self.query_hashes = {}
        self.hit_count = 0
        self.miss_count = 0

    def _hash_query(self, query: SearchRequest) -> str:
        """Generate hash for query"""
        query_dict = asdict(query)
        query_str = json.dumps(query_dict, sort_keys=True)
        return hashlib.md5(query_str.encode()).hexdigest()

    async def get(self, query: SearchRequest) -> Optional[List[SearchResult]]:
        """Get cached search results"""
        if not self.config.enabled:
            return None

        query_hash = self._hash_query(query)
        result = self.cache.get(query_hash)

        if result:
            self.hit_count += 1
            return result
        else:
            self.miss_count += 1
            return None

    async def put(self, query: SearchRequest, results: List[SearchResult]):
        """Cache search results"""
        if not self.config.enabled:
            return

        query_hash = self._hash_query(query)
        # Convert results to serializable format
        serialized_results = [result.to_dict() for result in results]
        self.cache.put(query_hash, serialized_results)

    def get_hit_rate(self) -> float:
        """Get cache hit rate"""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0


class IndexOptimizer:
    """Optimizes search indexes for better performance"""

    def __init__(self):
        self.index_stats = {}
        self.optimization_history = []

    async def analyze_index_performance(self, index_type: str, index_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze index performance and suggest optimizations"""
        analysis = {
            "index_type": index_type,
            "current_stats": index_stats,
            "recommendations": [],
            "performance_score": 0.0
        }

        # Analyze different metrics based on index type
        if index_type == "semantic":
            analysis = await self._analyze_semantic_index(index_stats, analysis)
        elif index_type == "keyword":
            analysis = await self._analyze_keyword_index(index_stats, analysis)
        elif index_type == "knowledge_graph":
            analysis = await self._analyze_graph_index(index_stats, analysis)

        self.index_stats[index_type] = analysis
        return analysis

    async def _analyze_semantic_index(self, stats: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze semantic search index"""
        doc_count = stats.get("document_count", 0)
        index_size = stats.get("index_size_mb", 0)
        avg_search_time = stats.get("avg_search_time", 0)

        # Performance scoring
        score = 100.0
        recommendations = []

        if avg_search_time > 0.5:  # 500ms threshold
            score -= 20
            recommendations.append("Consider increasing FAISS index size or using IVF index")

        if index_size > doc_count * 0.01:  # 10KB per document threshold
            score -= 15
            recommendations.append("Index size is large, consider compression or dimensionality reduction")

        if doc_count > 10000 and stats.get("index_type") == "Flat":
            score -= 25
            recommendations.append("Large dataset detected, consider upgrading to IVF or HNSW index")

        analysis["performance_score"] = max(0, score)
        analysis["recommendations"] = recommendations
        return analysis

    async def _analyze_keyword_index(self, stats: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze keyword search index"""
        doc_count = stats.get("document_count", 0)
        vocab_size = stats.get("vocabulary_size", 0)
        avg_search_time = stats.get("avg_search_time", 0)

        score = 100.0
        recommendations = []

        if vocab_size > doc_count * 0.8:  # Vocabulary too large
            score -= 15
            recommendations.append("Consider applying more aggressive stopword filtering")

        if avg_search_time > 0.1:  # 100ms threshold for keyword search
            score -= 20
            recommendations.append("Keyword search is slow, consider optimizing BM25 implementation")

        analysis["performance_score"] = max(0, score)
        analysis["recommendations"] = recommendations
        return analysis

    async def _analyze_graph_index(self, stats: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze knowledge graph index"""
        node_count = stats.get("node_count", 0)
        edge_count = stats.get("edge_count", 0)
        avg_traversal_time = stats.get("avg_traversal_time", 0)

        score = 100.0
        recommendations = []

        if edge_count > node_count * 10:  # Very dense graph
            score -= 20
            recommendations.append("Graph is very dense, consider filtering weak relationships")

        if avg_traversal_time > 0.2:  # 200ms threshold
            score -= 25
            recommendations.append("Graph traversal is slow, consider adding graph indexes")

        analysis["performance_score"] = max(0, score)
        analysis["recommendations"] = recommendations
        return analysis


class SearchOptimizer:
    """Main search optimization coordinator"""

    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.query_cache = QueryCache(self.config.cache_config)
        self.index_optimizer = IndexOptimizer()
        self.performance_metrics = PerformanceMetrics()
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        self.active_searches = {}
        self.search_history = defaultdict(list)

        # Performance monitoring
        self.query_times = []
        self.indexing_times = []
        self.error_log = []

    async def optimize_search_request(self, query: SearchRequest, search_func: Callable) -> List[SearchResult]:
        """Execute optimized search request"""
        start_time = time.time()
        search_id = str(hash(str(query)))[-8:]

        try:
            # Track concurrent queries
            self.performance_metrics.concurrent_queries += 1
            self.active_searches[search_id] = {
                "query": query.query,
                "start_time": start_time,
                "strategy": query.search_strategy
            }

            # Check cache first
            if self.config.enable_result_caching:
                cached_results = await self.query_cache.get(query)
                if cached_results:
                    # Convert back to SearchResult objects
                    results = [SearchResult.from_dict(result) for result in cached_results]
                    execution_time = time.time() - start_time
                    self._update_metrics(execution_time, cached_hit=True)
                    return results

            # Execute search with parallel processing if enabled
            if self.config.enable_parallel_search and query.search_strategy == "unified":
                results = await self._execute_parallel_search(query, search_func)
            else:
                results = await search_func(query)

            # Cache results
            if self.config.enable_result_caching and results:
                await self.query_cache.put(query, results)

            execution_time = time.time() - start_time
            self._update_metrics(execution_time, cached_hit=False)

            # Store in history
            self.search_history[query.search_strategy].append({
                "query": query.query,
                "results_count": len(results),
                "execution_time": execution_time,
                "timestamp": datetime.now()
            })

            return results

        except Exception as e:
            execution_time = time.time() - start_time
            self.performance_metrics.error_count += 1
            self.error_log.append({
                "error": str(e),
                "query": query.query,
                "timestamp": datetime.now(),
                "execution_time": execution_time
            })
            raise
        finally:
            # Clean up
            self.performance_metrics.concurrent_queries -= 1
            self.active_searches.pop(search_id, None)

    async def _execute_parallel_search(self, query: SearchRequest, search_func: Callable) -> List[SearchResult]:
        """Execute search components in parallel"""
        # This would require modifying the search function to support parallel execution
        # For now, just execute normally
        return await search_func(query)

    def _update_metrics(self, execution_time: float, cached_hit: bool):
        """Update performance metrics"""
        # Update query times
        self.query_times.append(execution_time)
        if len(self.query_times) > 1000:  # Keep last 1000 measurements
            self.query_times.pop(0)

        # Update cache metrics
        if cached_hit:
            self.performance_metrics.cache_hits += 1
        else:
            self.performance_metrics.cache_misses += 1

        # Update average query time
        if self.query_times:
            self.performance_metrics.avg_query_time = sum(self.query_times) / len(self.query_times)

        # Update total queries
        self.performance_metrics.total_queries += 1
        self.performance_metrics.last_updated = datetime.now()

    async def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        cache_hit_rate = self.query_cache.get_hit_rate()

        report = {
            "timestamp": datetime.now().isoformat(),
            "performance_metrics": asdict(self.performance_metrics),
            "cache_performance": {
                "hit_rate": cache_hit_rate,
                "cache_size": self.query_cache.cache.size(),
                "cache_enabled": self.config.cache_config.enabled
            },
            "optimization_config": asdict(self.config),
            "active_searches": len(self.active_searches),
            "search_history_summary": {
                strategy: {
                    "count": len(history),
                    "avg_time": sum(h["execution_time"] for h in history) / len(history) if history else 0,
                    "avg_results": sum(h["results_count"] for h in history) / len(history) if history else 0
                }
                for strategy, history in self.search_history.items()
            },
            "recent_errors": self.error_log[-10:] if self.error_log else [],
            "recommendations": self._generate_recommendations(cache_hit_rate)
        }

        return report

    def _generate_recommendations(self, cache_hit_rate: float) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []

        # Cache recommendations
        if cache_hit_rate < 0.3 and self.config.cache_config.enabled:
            recommendations.append("Cache hit rate is low, consider increasing cache TTL or size")

        # Performance recommendations
        if self.performance_metrics.avg_query_time > 1.0:
            recommendations.append("Average query time is high, consider enabling parallel search")

        if self.performance_metrics.concurrent_queries > 10:
            recommendations.append("High concurrent query load, consider increasing worker count")

        # Error rate recommendations
        if self.performance_metrics.error_count > 0:
            error_rate = self.performance_metrics.error_count / max(1, self.performance_metrics.total_queries)
            if error_rate > 0.05:  # 5% error rate
                recommendations.append("High error rate detected, review recent errors in the report")

        return recommendations

    async def clear_caches(self):
        """Clear all caches"""
        self.query_cache.cache.clear()
        self.query_cache.hit_count = 0
        self.query_cache.miss_count = 0

    async def update_config(self, new_config: OptimizationConfig):
        """Update optimization configuration"""
        self.config = new_config

        # Reinitialize cache with new config
        self.query_cache = QueryCache(new_config.cache_config)

        # Update thread pool
        self.executor.shutdown(wait=True)
        self.executor = ThreadPoolExecutor(max_workers=new_config.max_workers)

    async def cleanup(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)
        await self.clear_caches()


class PerformanceMonitor:
    """Real-time performance monitoring"""

    def __init__(self, optimizer: SearchOptimizer):
        self.optimizer = optimizer
        self.monitoring = False
        self.monitor_task = None

    async def start_monitoring(self, interval_seconds: int = 60):
        """Start performance monitoring"""
        self.monitoring = True
        self.monitor_task = asyncio.create_task(self._monitor_loop(interval_seconds))

    async def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass

    async def _monitor_loop(self, interval_seconds: int):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                # Update memory usage
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                cpu_percent = process.cpu_percent()

                self.optimizer.performance_metrics.memory_usage_mb = memory_mb
                self.optimizer.performance_metrics.cpu_usage_percent = cpu_percent

                # Log performance if needed
                if memory_mb > self.optimizer.config.cache_config.memory_limit_mb:
                    logging.warning(f"Memory usage high: {memory_mb:.1f}MB")

                await asyncio.sleep(interval_seconds)
            except Exception as e:
                logging.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(interval_seconds)


# Export main classes
__all__ = [
    "SearchOptimizer",
    "PerformanceMonitor",
    "OptimizationConfig",
    "CacheConfig",
    "PerformanceMetrics",
    "LRUCache",
    "QueryCache",
    "IndexOptimizer"
]