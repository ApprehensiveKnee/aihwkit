import torch

def check_gpu_status():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of available GPUs: {num_gpus}")
        for i in range(num_gpus):
            gpu = torch.cuda.get_device_properties(i)
            print(f"GPU {i}:")
            print(f"  Name: {gpu.name}")
            print(f"  CUDA Capability: {gpu.major}.{gpu.minor}")
            print(f"  Total Memory: {gpu.total_memory / 1024**2} MB")
            print(f"  Free Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**2} MB")
            print(f"  Is Busy: {torch.cuda.current_stream(i).query()}")
    else:
        print("No GPUs available.")

def get_free_gpu():
    for i in range(torch.cuda.device_count()):
        if not torch.cuda.current_stream(i).query():
            return i
    raise Exception("All GPUs are busy.")

def perform_test():
    try:
        gpu_id = get_free_gpu()
        gpu = torch.cuda.get_device_properties(gpu_id)
        print(f"Using GPU {gpu_id}: {gpu.name}")
        # Perform your test computation here
        # For example:
        x = torch.tensor([1, 2, 3]).cuda(gpu_id)
        y = x * 2
        print(f"Test result: {y}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == '__main__':
    check_gpu_status()
    perform_test()