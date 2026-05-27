"""
Profiling utilities for measuring performance of VLA models.
Includes timing and FLOPs calculation functionality.
"""

import time
import torch
from typing import Optional, Tuple, Any


class PerformanceTimer:
    """Context manager for timing code execution."""
    
    def __init__(self, name: str = "Operation", verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None
    
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.end_time = time.perf_counter()
        self.elapsed_time = self.end_time - self.start_time
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"[TIMING] {self.name}: {self.elapsed_time*1000:.2f} ms ({self.elapsed_time:.4f} s)")
            print(f"{'='*80}\n")
    
    def get_time(self) -> float:
        """Get elapsed time in seconds."""
        return self.elapsed_time if self.elapsed_time is not None else 0.0


def calculate_model_flops(
    model: torch.nn.Module,
    inputs: Tuple[torch.Tensor, ...],
    verbose: bool = True
) -> Optional[float]:
    """
    Calculate FLOPs for a model using thop library.
    
    Args:
        model: The model to profile
        inputs: Tuple of input tensors
        verbose: Whether to print results
    
    Returns:
        Total FLOPs (floating point operations)
    """
    try:
        from thop import profile, clever_format
        
        # Clone inputs to avoid modifying originals
        cloned_inputs = tuple(inp.clone() if isinstance(inp, torch.Tensor) else inp for inp in inputs)
        
        # Calculate FLOPs
        flops, params = profile(model, inputs=cloned_inputs, verbose=False)
        
        if verbose:
            flops_readable, params_readable = clever_format([flops, params], "%.3f")
            print(f"\n{'='*80}")
            print(f"[MODEL PROFILE]")
            print(f"  Total FLOPs: {flops_readable}")
            print(f"  Total Params: {params_readable}")
            print(f"  Raw FLOPs: {flops:.2e}")
            print(f"{'='*80}\n")
        
        return flops
    
    except ImportError:
        if verbose:
            print("[WARNING] thop library not available. Install with: pip install thop")
        return None
    except Exception as e:
        if verbose:
            print(f"[WARNING] Failed to calculate FLOPs: {e}")
        return None


def print_performance_summary(
    operation_name: str,
    time_ms: float,
    flops: Optional[float] = None
):
    """
    Print a formatted performance summary.
    
    Args:
        operation_name: Name of the operation being profiled
        time_ms: Time in milliseconds
        flops: Optional FLOPs count
    """
    print(f"\n{'='*80}")
    print(f"[PERFORMANCE SUMMARY] {operation_name}")
    print(f"  Execution Time: {time_ms:.2f} ms ({time_ms/1000:.4f} s)")
    
    if flops is not None:
        print(f"  Total FLOPs: {flops:.2e}")
        # Calculate FLOPS (FLOPs per second)
        if time_ms > 0:
            flops_per_sec = flops / (time_ms / 1000)
            print(f"  FLOPS: {flops_per_sec:.2e} FLOP/s")
            # Convert to GFLOPS for readability
            gflops = flops_per_sec / 1e9
            print(f"  GFLOPS: {gflops:.2f} GFLOP/s")
    
    print(f"{'='*80}\n")

