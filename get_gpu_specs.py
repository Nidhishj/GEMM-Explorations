import torch
import subprocess

def get_gpu_specs():
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return

    # 1. Basic properties from PyTorch
    props = torch.cuda.get_device_properties(0)
    name = props.name
    major = props.major
    minor = props.minor
    sm_count = props.multi_processor_count
    total_memory_gb = props.total_memory / (1024**3)

    # 2. Get Max SM Clock from nvidia-smi
    try:
        clock_str = subprocess.check_output(
            "nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits", 
            shell=True
        ).decode().strip()
        max_sm_clock_mhz = float(clock_str.split('\n')[0])
    except Exception as e:
        print("Could not get SM clock from nvidia-smi. Defaulting to 1500 MHz.")
        max_sm_clock_mhz = 1500.0

    # 3. Figure out FP32 cores per SM based on Compute Capability
    # Turing (7.5) = 64, Ampere (8.0, 8.6) = 64 or 128, Ada (8.9) = 128, Hopper (9.0) = 128
    cores_map = {
        (7, 0): (64, 8),
        (7, 5): (64, 8),
        (8, 0): (64, 4),
        (8, 6): (128, 4),
        (8, 9): (128, 4),
        (9, 0): (128, 4)
    }
    fp32_per_sm, tc_per_sm = cores_map.get((major, minor), (64, 0))
    
    total_fp32_cores = sm_count * fp32_per_sm
    
    # 4. Calculate Peak FP32 TFLOPs (FMA = 2 ops per clock)
    # TFLOPs = 2 * Cores * Clock (GHz)
    peak_flops = 2 * total_fp32_cores * (max_sm_clock_mhz / 1000.0) / 1000.0
    
    # 5. Get memory bandwidth from nvidia-smi (if not supported, hardcode common ones)
    try:
        mem_clock_str = subprocess.check_output(
            "nvidia-smi --query-gpu=clocks.max.memory --format=csv,noheader,nounits", 
            shell=True
        ).decode().strip()
        max_mem_clock_mhz = float(mem_clock_str.split('\n')[0])
    except Exception:
        max_mem_clock_mhz = None
    
    # We can't query bus width easily without NVML, so map common names to bus width
    bus_width_map = {
        "RTX 2080 Ti": 352,
        "RTX 3090": 384,
        "RTX 4090": 384,
        "A6000": 384,
        "A100": 5120, # HBM
    }
    
    bus_width = None
    for key in bus_width_map:
        if key in name:
            bus_width = bus_width_map[key]
            
    if bus_width and max_mem_clock_mhz:
        # Bandwidth formula for GDDR: (Bus Width / 8) * (Memory Clock * 2) / 1000 GB/s
        mem_bandwidth = (bus_width / 8.0) * (max_mem_clock_mhz * 2.0) / 1000.0
        bw_str = f"{mem_bandwidth:.2f} GB/s"
    else:
        bw_str = "N/A (Requires nvml or known bus width)"

    print("="*50)
    print(f"GPU Name:            {name}")
    print(f"Compute Capability:  {major}.{minor}")
    print(f"Global Memory:       {total_memory_gb:.2f} GB")
    print(f"SM Count:            {sm_count}")
    print(f"Max SM Clock:        {max_sm_clock_mhz} MHz")
    if max_mem_clock_mhz:
        print(f"Max Memory Clock:    {max_mem_clock_mhz} MHz")
    if bus_width:
        print(f"Memory Bus Width:    {bus_width}-bit")
    print(f"Peak Memory B/W:     {bw_str}")
    print(f"FP32 Cores / SM:     {fp32_per_sm} (Total: {total_fp32_cores})")
    print(f"Peak FP32 Compute:   {peak_flops:.2f} TFLOPs/s")
    print("="*50)

if __name__ == '__main__':
    get_gpu_specs()
