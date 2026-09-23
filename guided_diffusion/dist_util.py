"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 2

SETUP_RETRY_COUNT = 3

_use_mpi = None


def _check_mpi():
    """检测是否在分布式训练环境中运行，返回 True/False 并缓存结果。
    支持 MPI (mpirun) 和 torchrun 两种启动方式。
    通过环境变量判断，避免 import mpi4py 触发 MPI_Init() 导致卡死。"""
    global _use_mpi
    if _use_mpi is not None:
        return _use_mpi
    # 检查 torchrun 环境变量（torchrun 会设置这些变量）
    for varname in ["RANK", "LOCAL_RANK", "WORLD_SIZE"]:
        if varname in os.environ:
            _use_mpi = True
            return _use_mpi
    # 检查 MPI 环境变量，避免直接 import mpi4py（会触发 MPI_Init 卡死）
    for varname in ["PMI_RANK", "OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE"]:
        if varname in os.environ:
            _use_mpi = True
            return _use_mpi
    _use_mpi = False
    return _use_mpi


def _is_torchrun():
    """检查是否使用 torchrun 启动（torchrun 设置 RANK/LOCAL_RANK 但不设置 MPI 变量）。"""
    for varname in ["RANK", "LOCAL_RANK", "WORLD_SIZE"]:
        if varname in os.environ:
            # 检查是否也是 MPI 环境
            for mpi_var in ["PMI_RANK", "OMPI_COMM_WORLD_RANK"]:
                if mpi_var in os.environ:
                    return False
            return True
    return False


def setup_dist():
    """
    Setup a distributed process group.
    支持 MPI (mpirun) 和 torchrun 两种启动方式。
    """
    if dist.is_initialized():
        return
    
    if _is_torchrun():
        # torchrun 启动：环境变量已由 torchrun 设置，直接初始化
        backend = "gloo" if not th.cuda.is_available() else "nccl"
        dist.init_process_group(backend=backend, init_method="env://")
        # 关键：设置每个进程使用自己的 GPU
        if th.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            th.cuda.set_device(local_rank)
    elif _check_mpi():
        # MPI 启动：需要通过 MPI 设置环境变量
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        backend = "gloo" if not th.cuda.is_available() else "nccl"

        if backend == "gloo":
            hostname = "localhost"
        else:
            hostname = socket.gethostbyname(socket.getfqdn())
        os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
        os.environ["RANK"] = str(comm.rank)
        os.environ["WORLD_SIZE"] = str(comm.size)

        port = comm.bcast(_find_free_port(), root=0)
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group(backend=backend, init_method="env://")
        # MPI 模式也需要设置 GPU
        if th.cuda.is_available():
            local_rank = comm.rank % GPUS_PER_NODE
            th.cuda.set_device(local_rank)
    else:
        # 单卡/非分布式模式：跳过初始化
        if th.cuda.is_available():
            th.cuda.set_device(0)


def dev():
    """
    Get the device to use for torch.distributed.
    Uses LOCAL_RANK for distributed (torchrun), else current device.
    """
    if th.cuda.is_available():
        if _is_torchrun():
            local_rank = os.environ.get("LOCAL_RANK", "0")
            return th.device(f"cuda:{local_rank}")
        return th.device(f"cuda:{th.cuda.current_device()}")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file. 多卡时广播模型权重，单卡时直接加载。
    支持 torchrun 和 mpirun 两种分布式启动方式。
    """
    # 单卡/非分布式模式：直接加载
    if not _check_mpi() and not _is_torchrun():
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
        return th.load(io.BytesIO(data), **kwargs)
    
    if _is_torchrun():
        # torchrun 模式：使用 PyTorch 原生分布式广播
        if dist.get_rank() == 0:
            with bf.BlobFile(path, "rb") as f:
                data = f.read()
        else:
            data = None
        # 广播数据长度
        if dist.get_rank() == 0:
            data_len = [len(data)]
        else:
            data_len = [None]
        dist.broadcast_object_list(data_len)
        data_len = data_len[0]
        # 广播数据
        if dist.get_rank() == 0:
            tensor_list = [data]
        else:
            tensor_list = [b""]
        dist.broadcast_object_list(tensor_list)
        data = tensor_list[0]
        return th.load(io.BytesIO(data), **kwargs)
    else:
        # MPI 模式：使用 mpi4py 广播
        from mpi4py import MPI
        chunk_size = 2 ** 30  # MPI has a relatively small size limit
        if MPI.COMM_WORLD.Get_rank() == 0:
            with bf.BlobFile(path, "rb") as f:
                data = f.read()
            num_chunks = len(data) // chunk_size
            if len(data) % chunk_size:
                num_chunks += 1
            MPI.COMM_WORLD.bcast(num_chunks)
            for i in range(0, len(data), chunk_size):
                MPI.COMM_WORLD.bcast(data[i : i + chunk_size])
        else:
            num_chunks = MPI.COMM_WORLD.bcast(None)
            data = bytes()
            for _ in range(num_chunks):
                data += MPI.COMM_WORLD.bcast(None)

        return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    单卡模式或未初始化时跳过。
    """
    if not dist.is_initialized():
        return
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
