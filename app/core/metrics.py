"""
Prometheus metrics for monitoring translation service.
"""
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Info,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from typing import Optional
import time
import logging

logger = logging.getLogger(__name__)

# Request metrics
translation_requests_total = Counter(
    "nllb_translation_requests_total",
    "Total number of translation requests",
    ["source_lang", "target_lang", "status"],
)

translation_duration_seconds = Histogram(
    "nllb_translation_duration_seconds",
    "Time spent processing translations",
    ["source_lang", "target_lang"],
    buckets=(0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 30.0),
)

batch_translation_requests_total = Counter(
    "nllb_batch_translation_requests_total",
    "Total number of batch translation requests",
    ["status"],
)

batch_translation_size = Histogram(
    "nllb_batch_translation_size",
    "Number of texts in batch translations",
    buckets=(1, 5, 10, 20, 50, 100, 200, 500),
)

# Input/Output metrics
translation_input_length = Histogram(
    "nllb_translation_input_length_chars",
    "Length of input text in characters",
    ["source_lang"],
    buckets=(10, 50, 100, 250, 500, 1000, 2500, 5000),
)

translation_output_length = Histogram(
    "nllb_translation_output_length_chars",
    "Length of output text in characters",
    ["target_lang"],
    buckets=(10, 50, 100, 250, 500, 1000, 2500, 5000),
)

# Cache metrics
cache_hits_total = Counter(
    "nllb_cache_hits_total",
    "Total number of cache hits",
)

cache_misses_total = Counter(
    "nllb_cache_misses_total",
    "Total number of cache misses",
)

cache_size = Gauge(
    "nllb_cache_size",
    "Current number of items in cache",
)

# Model metrics
model_loaded = Gauge(
    "nllb_model_loaded",
    "Whether the model is loaded (1) or not (0)",
)

model_loading_duration_seconds = Histogram(
    "nllb_model_loading_duration_seconds",
    "Time spent loading the model",
)

gpu_available = Gauge(
    "nllb_gpu_available",
    "Whether GPU is available and being used (1) or not (0)",
)

# Error metrics
translation_errors_total = Counter(
    "nllb_translation_errors_total",
    "Total number of translation errors",
    ["error_type"],
)

# Health metrics
health_check_requests_total = Counter(
    "nllb_health_check_requests_total",
    "Total number of health check requests",
    ["status"],
)

# Rate limiting metrics
rate_limit_exceeded_total = Counter(
    "nllb_rate_limit_exceeded_total",
    "Total number of rate limit violations",
    ["endpoint"],
)

# Service info
service_info = Info(
    "nllb_service",
    "Information about the NLLB translation service",
)


class MetricsManager:
    """Manager for application metrics."""

    def __init__(self):
        """Initialize metrics manager."""
        self.start_time = time.time()
        logger.info("Metrics manager initialized")

    def record_translation(
        self,
        duration: float,
        source_lang: str,
        target_lang: str,
        input_length: int,
        output_length: int,
        success: bool = True,
        use_cache: bool = False,
    ) -> None:
        """
        Record metrics for a translation request.

        Args:
            duration: Translation duration in seconds
            source_lang: Source language code
            target_lang: Target language code
            input_length: Length of input text
            output_length: Length of output text
            success: Whether translation was successful
            use_cache: Whether result was from cache
        """
        status = "success" if success else "error"

        # Request metrics
        translation_requests_total.labels(
            source_lang=source_lang,
            target_lang=target_lang,
            status=status,
        ).inc()

        if success:
            # Duration metric
            translation_duration_seconds.labels(
                source_lang=source_lang,
                target_lang=target_lang,
            ).observe(duration)

            # Length metrics
            translation_input_length.labels(source_lang=source_lang).observe(input_length)
            translation_output_length.labels(target_lang=target_lang).observe(output_length)

        # Cache metrics
        if use_cache:
            cache_hits_total.inc()
        else:
            cache_misses_total.inc()

    def record_batch_translation(
        self,
        batch_size: int,
        success: bool = True,
    ) -> None:
        """
        Record metrics for a batch translation request.

        Args:
            batch_size: Number of texts in batch
            success: Whether translation was successful
        """
        status = "success" if success else "error"

        batch_translation_requests_total.labels(status=status).inc()

        if success:
            batch_translation_size.observe(batch_size)

    def record_error(self, error_type: str) -> None:
        """
        Record an error.

        Args:
            error_type: Type of error
        """
        translation_errors_total.labels(error_type=error_type).inc()

    def record_health_check(self, status: str = "healthy") -> None:
        """
        Record a health check request.

        Args:
            status: Health status
        """
        health_check_requests_total.labels(status=status).inc()

    def record_rate_limit_exceeded(self, endpoint: str) -> None:
        """
        Record a rate limit violation.

        Args:
            endpoint: Endpoint that was rate limited
        """
        rate_limit_exceeded_total.labels(endpoint=endpoint).inc()

    def update_model_status(self, loaded: bool, gpu: bool) -> None:
        """
        Update model status metrics.

        Args:
            loaded: Whether model is loaded
            gpu: Whether GPU is being used
        """
        model_loaded.set(1 if loaded else 0)
        gpu_available.set(1 if gpu else 0)

    def update_cache_size(self, size: int) -> None:
        """
        Update cache size metric.

        Args:
            size: Current cache size
        """
        cache_size.set(size)

    def record_model_loading(self, duration: float) -> None:
        """
        Record model loading time.

        Args:
            duration: Loading duration in seconds
        """
        model_loading_duration_seconds.observe(duration)

    def set_service_info(
        self,
        version: str,
        model_name: str,
        device: str,
    ) -> None:
        """
        Set service information.

        Args:
            version: Service version
            model_name: Model name
            device: Device being used
        """
        service_info.info({
            "version": version,
            "model": model_name,
            "device": device,
        })

    def get_metrics(self) -> bytes:
        """
        Get Prometheus metrics in text format.

        Returns:
            Metrics in Prometheus text format
        """
        return generate_latest()


# Global metrics manager instance
_metrics_manager: Optional[MetricsManager] = None


def get_metrics_manager() -> MetricsManager:
    """Get or create metrics manager singleton."""
    global _metrics_manager
    if _metrics_manager is None:
        _metrics_manager = MetricsManager()
    return _metrics_manager


class MetricsMiddleware:
    """FastAPI middleware for automatic metrics recording."""

    def __init__(self, app, metrics_manager: Optional[MetricsManager] = None):
        """
        Initialize metrics middleware.

        Args:
            app: FastAPI app instance
            metrics_manager: Metrics manager instance
        """
        self.app = app
        self.metrics_manager = metrics_manager or get_metrics_manager()

    async def __call__(self, scope, receive, send):
        """Process request and record metrics."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Track request timing
        start_time = time.time()

        # Continue with request
        await self.app(scope, receive, send)

        # Record metrics
        duration = time.time() - start_time
        path = scope.get("path", "")

        # Record specific metrics based on endpoint
        if path.endswith("/health"):
            self.metrics_manager.record_health_check()
