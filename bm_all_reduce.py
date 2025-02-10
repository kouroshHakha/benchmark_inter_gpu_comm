#!/usr/bin/env python
import argparse
import os
import time
import socket


import torch
import torch.distributed as dist
import ray
from ray.util.placement_group import placement_group


def get_free_port():
    """Find and return a free port on the current node."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return str(port)

def benchmark_all_reduce(tensor, iters, warmup):
    # Warm-up iterations.

    for _ in range(warmup):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()

    # Benchmark iterations.
    start = time.time()
    for _ in range(iters):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    end = time.time()

    avg_time = (end - start) / iters
    # For all-reduce, we assume that each element's data is effectively "sent" twice.
    world_size = dist.get_world_size()
    all_reduce_multiplier = 2 * (world_size - 1) / world_size
    num_bytes = tensor.numel() * tensor.element_size() * all_reduce_multiplier
    bandwidth = num_bytes / avg_time / (1024 ** 3)  # in GB/s
    return avg_time, bandwidth

@ray.remote(num_gpus=1)
def run_all_reduce(rank, world_size, master_addr, master_port, tensor_size, iters, warmup):
    # Set environment variables for torch.distributed.
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)

    # Initialize the process group using NCCL.
    dist.init_process_group(backend="nccl")
    # In a Ray task with num_gpus=1, torch.cuda.current_device() should reflect the assigned GPU.
    torch.cuda.set_device(torch.cuda.current_device())

    # Create a random tensor on the assigned GPU.
    bytes_per_element = torch.tensor([], dtype=torch.bfloat16).element_size()
    num_elements = tensor_size // bytes_per_element
    tensor = torch.randn(num_elements, device="cuda")

    # Run the all-reduce benchmark.
    avg_time, bandwidth = benchmark_all_reduce(tensor, iters, warmup)


    dist.destroy_process_group()
    return avg_time, bandwidth

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ray-wrapped PyTorch NCCL All-Reduce Benchmark with Placement Group"
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=16,
        help="Total number of processes (default: 16)",
    )
    parser.add_argument(
        "--tensor_size",
        type=int,
        default=256 * 1024 * 1024,
        help="Number of elements in the tensor (default: 256M)",
    )
    parser.add_argument(
        "--iters", 
        type=int, 
        default=20, 
        help="Number of benchmark iterations (default: 20)"
    )
    parser.add_argument(
        "--warmup", 
        type=int, 
        default=5, 
        help="Number of warmup iterations (default: 5)"
    )
    args = parser.parse_args()


    # (Optional) if using IB, set NCCL_IB_DISABLE=0; here we leave it as default.
    ray.init(runtime_env={
        'py_executable': 'uv run --isolated --directory ./benchmark_inter_gpu_comm',
        'working_dir': '/home/ray/default',
        'env_vars': {
            # 'NCCL_DEBUG': 'INFO',
            # 'NCCL_IB_HCA': 'mlx5_15, mlx5_17',
            # 'NCCL_NTHREADS': '8',
            # 'NCCL_P2P_LEVEL': 'SYS',
            'NCCL_IB_DISABLE': '1',
        }
    })

    # Get the master address automatically using Ray's API
    master_addr = ray.util.get_node_ip_address()
    # Get a free port
    master_port = get_free_port()

    print(f"Using master address: {master_addr}, port: {master_port}")


    # Create a placement group with one bundle per process.
    # Each bundle requests 1 CPU and 1 GPU.
    pg = placement_group([{"CPU": 1, "GPU": 1} for _ in range(args.world_size)],
                         strategy="PACK")
    # Wait for the placement group to be ready.
    ray.get(pg.ready())

    # Launch one remote task per process, scheduling them on the placement group.
    tasks = [
        run_all_reduce.options(placement_group=pg).remote(
            rank,
            args.world_size,
            master_addr,
            master_port,
            args.tensor_size,
            args.iters,
            args.warmup,
        )
        for rank in range(args.world_size)
    ]

    results = ray.get(tasks)
    avg_times = [res[0] for res in results]
    avg_bandwidths = [res[1] for res in results]

    overall_avg_time = sum(avg_times) / len(avg_times)
    overall_avg_bw = sum(avg_bandwidths) / len(avg_bandwidths)

    print(
        f"Overall average time: {overall_avg_time * 1e6:.2f} us, Overall average bandwidth: {overall_avg_bw:.2f} GB/s"
    )

