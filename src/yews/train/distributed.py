import multiprocessing
import os
import random
import signal
import threading
import traceback
from typing import Callable, List, Optional
import torch


def is_master_proc() -> bool:
    """Determines if the current process is the master process.

    Master process is responsible for logging, writing and loading checkpoints. In
    the multi GPU setting, we assign the master role to the rank 0 process. When
    training using a single GPU, there is a single process which is considered master.
    """
    return torch.distributed.get_rank() == 0


def init_process_group(
    proc_rank: int, world_size: int, host: str, port: int, backend: str
) -> None:
    """Initializes the default process group."""
    # Set the GPU to use
    err_msg = "Cannot use more GPU devices than available."
    if proc_rank >= torch.cuda.device_count():
        raise ValueError(err_msg)
    torch.cuda.set_device(proc_rank)
    # Initialize the process group
    torch.distributed.init_process_group(
        backend=backend,
        init_method="tcp://{}:{}".format(host, port),
        world_size=world_size,
        rank=proc_rank,
    )


def destroy_process_group() -> None:
    """Destroys the default process group."""
    torch.distributed.destroy_process_group()


def all_reduce(tensors: List[torch.Tensor]) -> List[torch.Tensor]:
    """Performs the scaled all_reduce operation on the provided tensors.

    The input tensors are modified in-place. Currently supports only the sum
    reduction operator. The reduced values needs to be scaled by the number of
    GPUs outside this function.
    """

    # Queue the reductions
    reductions = []
    for tensor in tensors:
        reduction = torch.distributed.all_reduce(tensor, async_op=True)
        reductions.append(reduction)
    # Wait for reductions to finish
    for reduction in reductions:
        reduction.wait()
    # Scale the results
    return tensors


class ChildException(Exception):
    """Wraps an exception from a child process."""


class ErrorHandler:
    """Multiprocessing error handler (based on fairseq's).

    Listens for errors in child processes and propagates the tracebacks to the parent.
    """

    def __init__(self, error_queue):
        # Shared error queue
        self.error_queue = error_queue
        # Children processes sharing the error queue
        self.children_pids = []
        # Start a thread listening to errors
        self.error_listener = threading.Thread(target=self.listen, daemon=True)
        self.error_listener.start()
        # Register the signal handler
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        """Registers a child process."""
        self.children_pids.append(pid)

    def listen(self):
        """Listens for errors in the error queue."""
        # Wait until there is an error in the queue
        child_trace = self.error_queue.get()
        # Put the error back for the signal handler
        self.error_queue.put(child_trace)
        # Invoke the signal handler
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, _sig_num, _stack_frame):
        """Signal handler."""
        # Kill children processes
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)
        # Propagate the error from the child process
        raise ChildException(self.error_queue.get())


def run(
    proc_rank: int,
    world_size: int,
    host: str,
    port: int,
    backend: str,
    error_queue,
    fun: Callable,
    fun_args: tuple,
    fun_kwargs: dict,
) -> None:
    """Runs a function from a child process."""
    try:
        # Initialize the process group
        init_process_group(proc_rank, world_size, host, port, backend)
        # Run the function
        fun(*fun_args, **fun_kwargs)
    except KeyboardInterrupt:
        # Killed by the parent process
        pass
    except Exception:
        # Propagate exception to the parent process
        error_queue.put(traceback.format_exc())
    finally:
        # Destroy the process group
        destroy_process_group()


def multi_proc_run(
    num_proc: int,
    backend: str,
    host: str,
    port_range: List[int],
    fun: Callable,
    fun_args: tuple = (),
    fun_kwargs: Optional[dict] = None,
):
    """Runs a function in a multi-proc setting (unless num_proc == 1)."""
    # There is no need for multi-proc in the single-proc case
    fun_kwargs = fun_kwargs if fun_kwargs else {}
    if num_proc == 1:
        fun(*fun_args, **fun_kwargs)
        return
    # Handle errors from training subprocesses
    error_queue = multiprocessing.SimpleQueue()
    error_handler = ErrorHandler(error_queue)
    # Get a random port to use (without using global random number generator)
    port = random.Random().randint(port_range[0], port_range[1])
    # Run each training subprocess
    ps = []
    for i in range(num_proc):
        p_i = multiprocessing.Process(
            target=run,
            args=(
                i,
                num_proc,
                host,
                port,
                backend,
                error_queue,
                fun,
                fun_args,
                fun_kwargs,
            ),
        )
        ps.append(p_i)
        p_i.start()
        error_handler.add_child(p_i.pid)
    # Wait for each subprocess to finish
    for p in ps:
        p.join()
