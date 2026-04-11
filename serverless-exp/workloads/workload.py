import time
import os
import json
import sys

# Force output flushing
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

def get_memory_limit():
    # Detect cgroup memory limit to simulate "Parkinson's Law"
    # (App expanding to use available RAM)
    try:
        with open("/sys/fs/cgroup/memory/memory.limit_in_bytes", "r") as f:
            limit = int(f.read())
            # Cap at 1GB for safety if no limit is set
            return min(limit, 1024 * 1024 * 1024)
    except:
        return 1024 * 1024 * 1024 # Default to 1GB

def handle(event, context):
    if os.environ.get('PROFILE_MODE') == '1':
        time.sleep(3.0) 
    
    if isinstance(event, str):
        try: event = json.loads(event)
        except: event = {"workload": event}

    mode = event.get('workload', 'cpu')
    print(f"DEBUG: Starting MEMORY-EXPANSION workload '{mode}'...", file=sys.stderr)
    
    start_time = time.time()
    TARGET_DURATION = 10.0 

    # Determine allocation size: 50% of Container Limit
    # 128MB Container -> Allocates 64MB (Efficient)
    # 1024MB Container -> Allocates 512MB (Wasteful)
    mem_limit = get_memory_limit()
    alloc_size = int(mem_limit * 0.5)
    
    print(f"DEBUG: Allocating {alloc_size / 1024 / 1024:.2f} MB", file=sys.stderr)

    # --- THE WORKLOAD ---
    # Rapidly allocate and release memory to simulate 
    # garbage collection overhead and page zeroing costs.
    loops = 0
    while (time.time() - start_time) < TARGET_DURATION:
        # Allocate massive array (forces OS to zero pages)
        # This burns ENERGY proportional to size.
        data = bytearray(alloc_size)
        
        # Touch data to ensure it's actually allocated in RAM
        data[0] = 1
        data[-1] = 1
        
        # Explicit delete to force churn
        del data
        
        # Tiny sleep to let heat dissipate
        time.sleep(0.05)
        loops += 1

    print(f"DEBUG: Expansion Done ({loops} loops)", file=sys.stderr)
    return "Expansion Done"

if __name__ == "__main__":
    if len(sys.argv) > 1:
        handle(sys.argv[1], {})
