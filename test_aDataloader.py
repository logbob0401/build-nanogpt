import torch
import time
from torch.utils.data import DataLoader
import pytest
from torch.utils.benchmark import Timer
from aDataloader import AsyncDataLoaderConfig,AsyncDataLoader

def test_async_dataloader():
    """Test the async data loader implementation"""
    
    # Configuration
    config = AsyncDataLoaderConfig(
        batch_size=64,
        seq_length=1024,
        process_rank=0,
        num_processes=1,
        prefetch_factor=3,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Create loader
    loader = AsyncDataLoader(config)
    
    def benchmark_batch_loading():
        """Benchmark batch loading performance"""
        start = time.time()
        num_batches = 100
        
        # Warmup
        for _ in range(5):
            x, y = loader.next_batch()
            
        # Timing
        times = []
        for i in range(num_batches):
            batch_start = time.time()
            x, y = loader.next_batch()
            times.append(time.time() - batch_start)
            
            # Basic checks
            assert x.shape == (config.batch_size, config.seq_length)
            assert y.shape == (config.batch_size, config.seq_length)
            assert x.device == torch.device(config.device)
            assert y.device == torch.device(config.device)
            
        end = time.time()
        
        # Calculate metrics
        avg_time = sum(times) / len(times)
        throughput = num_batches / (end - start)
        return avg_time, throughput
        
    def test_reset():
        """Test loader reset functionality"""
        # Get initial batch
        x1, y1 = loader.next_batch()
        
        # Reset loader
        loader.reset()
        
        # Get new batch
        x2, y2 = loader.next_batch()
        
        # Should get same data after reset
        assert torch.equal(x1, x2)
        assert torch.equal(y1, y2)
        
    def compare_with_basic_loader():
        """Compare with basic DataLoader"""
        # Setup basic loader
        basic_config = AsyncDataLoaderConfig(
            batch_size=64,
            seq_length=1024,
            process_rank=0,
            num_processes=1,
            device=config.device
        )
        
        def time_basic_loader():
            loader = DataLoader(
                range(1000),
                batch_size=basic_config.batch_size,
                num_workers=2
            )
            for batch in loader:
                batch = torch.tensor(batch).to(basic_config.device)
                
        def time_async_loader():
            loader = AsyncDataLoader(config)
            for _ in range(100):
                x, y = loader.next_batch()
                
        # Benchmark both
        basic_timer = Timer(
            stmt="time_basic_loader()",
            globals={'time_basic_loader': time_basic_loader}
        )
        
        async_timer = Timer(
            stmt="time_async_loader()",
            globals={'time_async_loader': time_async_loader}
        )
        
        basic_time = basic_timer.timeit(100)
        async_time = async_timer.timeit(100)
        
        return basic_time, async_time
        
    # Run tests
    print("\nRunning async dataloader tests...")
    
    # Test basic functionality
    avg_time, throughput = benchmark_batch_loading()
    print(f"\nBatch loading performance:")
    print(f"Average batch time: {avg_time*1000:.2f}ms")
    print(f"Throughput: {throughput:.2f} batches/sec")
    
    # Test reset
    test_reset()
    print("\nReset functionality: PASSED")
    
    # Compare with basic loader
    basic_time, async_time = compare_with_basic_loader()
    print("\nLoader comparison:")
    print(f"Basic loader time: {basic_time.mean*1000:.2f}ms")
    print(f"Async loader time: {async_time.mean*1000:.2f}ms")
    print(f"Speedup: {basic_time.mean/async_time.mean:.2f}x")
    
    # Get detailed metrics
    metrics = loader.get_performance_metrics()
    print("\nDetailed performance metrics:")
    for k, v in metrics.items():
        if 'time' in k:
            print(f"{k}: {v*1000:.2f}ms")
        else:
            print(f"{k}: {v}")
            
if __name__ == "__main__":
    test_async_dataloader()
