import numpy as np
import time
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import psutil

# Constants
MATRIX_SIZE = 5000 
NUM_MATRICES = 500  
THREAD_COUNTS = [1, 2, 3, 4, 5, 6, 7, 8]  
BATCH_SIZE = 1  
DTYPE = np.float32  

def multiply_batch(batch_indices, constant_matrix, results):
    for idx in batch_indices:
        try:
            matrix = np.random.rand(MATRIX_SIZE, MATRIX_SIZE).astype(DTYPE) 
            results[idx] = np.dot(matrix, constant_matrix)
        except MemoryError as e:
            print(f"MemoryError: {e}")
            results[idx] = None

def monitor_cpu_usage(interval, duration):
    cpu_usage = []
    for _ in range(int(duration / interval)):
        cpu_usage.append(psutil.cpu_percent(interval=interval))
    return cpu_usage

execution_times = []
cpu_usages = {}

for threads in THREAD_COUNTS:
    constant_matrix = np.random.rand(MATRIX_SIZE, MATRIX_SIZE).astype(DTYPE) 
    results = [None] * NUM_MATRICES

    start_time = time.time()
    try:
        with ThreadPoolExecutor(max_workers=threads) as executor:
            batch_indices = list(range(NUM_MATRICES))
            futures = []
            for i in range(0, len(batch_indices), BATCH_SIZE):
                chunk = batch_indices[i : i + BATCH_SIZE]
                futures.append(executor.submit(multiply_batch, chunk, constant_matrix, results))
            for future in futures:
                future.result() 
    except Exception as e:
        print(f"Error during thread execution: {e}")
        execution_times.append(None) 
        continue

    end_time = time.time()
    execution_times.append(end_time - start_time)

    cpu_usages[threads] = monitor_cpu_usage(interval=0.5, duration=end_time - start_time)

execution_times = [t for t in execution_times if t is not None]

plt.figure(figsize=(10, 5))
plt.plot(THREAD_COUNTS[:len(execution_times)], execution_times, marker='o', label="Execution Time")
plt.title('Execution Time vs Number of Threads (8GB RAM)')
plt.xlabel('Number of Threads')
plt.ylabel('Execution Time (seconds)')
plt.legend()
plt.grid()
plt.show()

for threads, usage in cpu_usages.items():
    print(f"Threads: {threads}, Avg CPU Usage: {np.mean(usage):.2f}%")

plt.figure(figsize=(10, 5))
for threads, usage in cpu_usages.items():
    plt.plot(range(len(usage)), usage, label=f"{threads} Threads")
plt.title("CPU Usage vs Time")
plt.xlabel("Time (steps)")
plt.ylabel("CPU Usage (%)")
plt.legend()
plt.grid()
plt.show()
