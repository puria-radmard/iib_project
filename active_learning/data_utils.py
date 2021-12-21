import subprocess, torch
from numpy import ndarray

def total_sum(thing):
    if isinstance(thing, set):
        return sum(thing)
    elif isinstance(thing, list):
        first_sum = sum(thing)
        if isinstance(first_sum, int) or isinstance(first_sum, float):
            return first_sum
        else:
            return sum(first_sum)
    elif isinstance(thing, ndarray):
        return sum(thing)
    elif isinstance(thing, bool):
        return 1 if thing else 0
    elif isinstance(thing, torch.Tensor):
        return thing.sum().item()
    
def command_line(command):
    return subprocess.check_output(command.split())