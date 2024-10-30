import os
import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
import threading
from queue import Queue
import time
import numpy as np
from contextlib import contextmanager
from typing import Optional, Tuple
import logging
from dataclasses import dataclass
import torch.cuda.nvtx as nvtx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AsyncDataLoaderConfig:
    """Configuration for AsyncDataLoader"""
    batch_size: int
    seq_length: int
    process_rank: int
    num_processes: int
    prefetch_factor: int = 3
    pin_memory: bool = True
    num_workers: int = 2
    device: str = "cuda"

class PrefetchQueue:
    """Thread-safe queue with CUDA streams management"""
    def __init__(self, maxsize: int, device: str):
        self.queue = Queue(maxsize=maxsize)
        self.device = device
        self.streams = []
        if device.startswith("cuda"):
            self.streams = [torch.cuda.Stream() for _ in range(maxsize)]
        self.stream_pool = Queue(maxsize=maxsize)
        for stream in self.streams:
            self.stream_pool.put(stream)

    def get(self) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.cuda.Stream]]:
        item = self.queue.get()
        if isinstance(item, tuple) and len(item) == 3:
            return item
        return item[0], item[1], None

    def put(self, item: Tuple[torch.Tensor, torch.Tensor], stream: Optional[torch.cuda.Stream] = None):
        self.queue.put((item[0], item[1], stream))

    @contextmanager
    def get_stream(self):
        stream = self.stream_pool.get() if self.streams else None
        try:
            yield stream
        finally:
            if stream is not None:
                self.stream_pool.put(stream)

class AsyncDataLoader:
    """
    Asynchronous data loader with GPU prefetching and performance monitoring
    """
    def __init__(self, config: AsyncDataLoaderConfig):
        self.config = config
        self.device = config.device
        
        # Initialize queues and events
        self.data_queue = PrefetchQueue(config.prefetch_factor, config.device)
        self.should_stop = threading.Event()
        
        # Create streams for async operations
        if self.device.startswith("cuda"):
            self.main_stream = torch.cuda.current_stream()
        
        # Performance metrics
        self.total_batches = 0
        self.total_time = 0
        self.prefetch_times = []
        self.transfer_times = []
        
        # Initialize data source
        self._init_data_source()
        
        # Start prefetch thread
        self.start_prefetch()

    def _init_data_source(self):
        """Initialize the data source and loading state"""
        data_root = "edu_fineweb10B"
        self.shards = self._get_shards(data_root)
        self.current_shard = 0
        self.tokens = self._load_tokens(self.shards[self.current_shard])
        self.current_position = (self.config.batch_size * 
                               self.config.seq_length * 
                               self.config.process_rank)

    def _get_shards(self, data_root: str) -> list:
        """Get list of data shards"""
        shards = [f for f in os.listdir(data_root) if f.endswith('.npy')]
        shards.sort()
        return [os.path.join(data_root, s) for s in shards]

    def _load_tokens(self, filename: str) -> torch.Tensor:
        """Load and preprocess tokens from file"""
        nvtx.range_push(f"load_tokens:{filename}")
        try:
            npt = np.load(filename)
            npt = npt.astype(np.int32)
            if self.config.pin_memory:
                ptt = torch.tensor(npt, dtype=torch.long).pin_memory()
            else:
                ptt = torch.tensor(npt, dtype=torch.long)
            return ptt
        finally:
            nvtx.range_pop()

    def _prefetch_worker(self):
        """Background thread that loads and moves data to GPU"""
        while not self.should_stop.is_set():
            try:
                nvtx.range_push("prefetch_batch")
                start_time = time.time()
                
                # Get batch data
                batch = self._get_next_batch()
                if batch is None:
                    continue
                    
                x, y = batch
                
                # Get stream for async transfer
                with self.data_queue.get_stream() as stream:
                    if stream is not None:
                        with torch.cuda.stream(stream):
                            # Async transfer to GPU
                            x_device = x.to(self.device, non_blocking=True)
                            y_device = y.to(self.device, non_blocking=True)
                            
                            # Record transfer time
                            transfer_time = time.time() - start_time
                            self.transfer_times.append(transfer_time)
                            
                            # Put in queue
                            self.data_queue.put((x_device, y_device), stream)
                    else:
                        # CPU case or no stream available
                        self.data_queue.put((x, y))
                        
                # Record prefetch time
                prefetch_time = time.time() - start_time
                self.prefetch_times.append(prefetch_time)
                
            except Exception as e:
                logger.error(f"Error in prefetch worker: {e}")
                break
            finally:
                nvtx.range_pop()

    def _get_next_batch(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get next batch from current shard or load new shard if needed"""
        B, T = self.config.batch_size, self.config.seq_length
        
        # Check if we need to load next shard
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = self._load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.config.process_rank
            
        # Extract batch
        buf = self.tokens[self.current_position:self.current_position + B * T + 1]
        if len(buf) < B * T + 1:
            return None
            
        # Process batch
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        
        # Update position
        self.current_position += B * T * self.config.num_processes
        
        return x, y

    def start_prefetch(self):
        """Start the prefetch thread"""
        self.load_thread = threading.Thread(target=self._prefetch_worker)
        self.load_thread.daemon = True
        self.load_thread.start()

    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get next batch, measuring timing"""
        start_time = time.time()
        
        # Get batch from queue
        x, y, stream = self.data_queue.get()
        
        # Synchronize if needed
        if stream is not None:
            stream.synchronize()
            
        # Update metrics
        batch_time = time.time() - start_time
        self.total_time += batch_time
        self.total_batches += 1
        
        return x, y

    def get_performance_metrics(self) -> dict:
        """Get loading performance metrics"""
        metrics = {
            "avg_batch_time": self.total_time / max(1, self.total_batches),
            "avg_prefetch_time": np.mean(self.prefetch_times) if self.prefetch_times else 0,
            "avg_transfer_time": np.mean(self.transfer_times) if self.transfer_times else 0,
            "total_batches": self.total_batches
        }
        return metrics

    def reset(self):
        """Reset the data loader state"""
        # Stop current prefetch
        self.should_stop.set()
        if hasattr(self, 'load_thread'):
            self.load_thread.join()
        self.should_stop.clear()
        
        # Clear queue
        while True:
            try:
                self.data_queue.get(timeout=0.1)
            except:
                break
                
        # Reset state
        self.current_shard = 0
        self.tokens = self._load_tokens(self.shards[self.current_shard])
        self.current_position = (self.config.batch_size * 
                               self.config.seq_length * 
                               self.config.process_rank)
        
        # Reset metrics
        self.total_batches = 0
        self.total_time = 0
        self.prefetch_times = []
        self.transfer_times = []
        
        # Restart prefetch
        self.start_prefetch()

    def __del__(self):
        """Cleanup"""
        self.should_stop.set()
        if hasattr(self, 'load_thread'):
            self.load_thread.join()
            