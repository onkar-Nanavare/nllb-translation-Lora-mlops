"""
Parallel batch processing for translations.
Optimizes throughput by processing multiple translations concurrently.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Tuple
import time
import logging

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Handles parallel batch processing of translation requests.
    """

    def __init__(
        self,
        model: AutoModelForSeq2SeqLM,
        tokenizer: AutoTokenizer,
        device: str = "cuda",
        max_batch_size: int = 32,
        max_workers: int = 4,
        timeout: float = 30.0,
    ):
        """
        Initialize batch processor.

        Args:
            model: Translation model
            tokenizer: Model tokenizer
            device: Device to use (cuda/cpu/mps)
            max_batch_size: Maximum batch size for model inference
            max_workers: Number of worker threads
            timeout: Timeout for batch processing
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_batch_size = max_batch_size
        self.max_workers = max_workers
        self.timeout = timeout
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def _process_batch(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        max_length: int = 512,
    ) -> List[str]:
        """
        Process a single batch of texts.

        Args:
            texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code
            max_length: Maximum sequence length

        Returns:
            List of translated texts
        """
        try:
            # Set source language
            self.tokenizer.src_lang = source_lang

            # Tokenize all texts at once
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate translations
            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **inputs,
                    forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(target_lang) if not hasattr(self.tokenizer, 'lang_code_to_id') else self.tokenizer.lang_code_to_id[target_lang],
                    max_length=max_length,
                    num_beams=5,
                    early_stopping=True,
                )

            # Decode all outputs
            translations = self.tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )

            return translations

        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")
            raise

    def process_parallel(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        max_length: int = 512,
    ) -> List[str]:
        """
        Process texts in parallel batches.

        Args:
            texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code
            max_length: Maximum sequence length

        Returns:
            List of translated texts in original order
        """
        if not texts:
            return []

        start_time = time.time()
        num_texts = len(texts)

        logger.info(f"Processing {num_texts} texts in parallel (batch_size={self.max_batch_size})")

        # Split into batches
        batches = [
            texts[i:i + self.max_batch_size]
            for i in range(0, len(texts), self.max_batch_size)
        ]

        logger.debug(f"Split into {len(batches)} batches")

        # Process batches sequentially (model is not thread-safe)
        # But within each batch, we process multiple texts together
        all_translations = []
        for batch_idx, batch in enumerate(batches):
            logger.debug(f"Processing batch {batch_idx + 1}/{len(batches)}")
            translations = self._process_batch(
                batch, source_lang, target_lang, max_length
            )
            all_translations.extend(translations)

        elapsed_time = time.time() - start_time
        logger.info(
            f"Batch processing complete: {num_texts} texts in {elapsed_time:.2f}s "
            f"({num_texts / elapsed_time:.2f} texts/sec)"
        )

        return all_translations

    async def process_async(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        max_length: int = 512,
    ) -> List[str]:
        """
        Process texts asynchronously.

        Args:
            texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code
            max_length: Maximum sequence length

        Returns:
            List of translated texts
        """
        loop = asyncio.get_event_loop()

        # Run in thread pool to avoid blocking
        result = await loop.run_in_executor(
            self.executor,
            self.process_parallel,
            texts,
            source_lang,
            target_lang,
            max_length,
        )

        return result

    def shutdown(self):
        """Shutdown the thread pool executor."""
        logger.info("Shutting down batch processor")
        self.executor.shutdown(wait=True)


class DynamicBatchProcessor(BatchProcessor):
    """
    Advanced batch processor with dynamic batching.
    Groups requests dynamically based on arrival time.
    """

    def __init__(
        self,
        model: AutoModelForSeq2SeqLM,
        tokenizer: AutoTokenizer,
        device: str = "cuda",
        max_batch_size: int = 32,
        max_workers: int = 4,
        timeout: float = 30.0,
        batch_timeout_ms: float = 100.0,  # Wait time to collect batch
    ):
        """
        Initialize dynamic batch processor.

        Args:
            batch_timeout_ms: Maximum time to wait for batch to fill (milliseconds)
        """
        super().__init__(model, tokenizer, device, max_batch_size, max_workers, timeout)
        self.batch_timeout_ms = batch_timeout_ms
        self.pending_requests: List[Tuple[str, asyncio.Future]] = []
        self.lock = asyncio.Lock()

    async def process_request(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        max_length: int = 512,
    ) -> str:
        """
        Process a single request with dynamic batching.

        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            max_length: Maximum sequence length

        Returns:
            Translated text
        """
        # Create a future for this request
        future = asyncio.Future()

        async with self.lock:
            self.pending_requests.append((text, source_lang, target_lang, future))

            # If batch is full, process immediately
            if len(self.pending_requests) >= self.max_batch_size:
                await self._process_pending_batch(max_length)

        # Wait for batch timeout
        try:
            result = await asyncio.wait_for(
                future,
                timeout=self.batch_timeout_ms / 1000.0
            )
            return result
        except asyncio.TimeoutError:
            # Process whatever we have
            async with self.lock:
                if self.pending_requests:
                    await self._process_pending_batch(max_length)
            return await future

    async def _process_pending_batch(self, max_length: int):
        """Process all pending requests as a batch."""
        if not self.pending_requests:
            return

        requests = self.pending_requests.copy()
        self.pending_requests.clear()

        texts = [req[0] for req in requests]
        source_lang = requests[0][1]  # Assume same language for batch
        target_lang = requests[0][2]
        futures = [req[3] for req in requests]

        try:
            # Process batch
            translations = await self.process_async(
                texts, source_lang, target_lang, max_length
            )

            # Set results
            for future, translation in zip(futures, translations):
                if not future.done():
                    future.set_result(translation)

        except Exception as e:
            # Set exception for all futures
            for future in futures:
                if not future.done():
                    future.set_exception(e)
