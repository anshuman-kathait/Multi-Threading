import numpy as np
import time
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import psutil

MATRIX_SIZE = 5000 
NUM_MATRICES = 500  
THREAD_COUNTS = [1, 2, 3, 4, 5, 6, 7, 8] 
BATCH_SIZE = 2 

def multiply_batch(batch_indices, constant_matrix, results):
    for idx in batch_indices:
        matrix = np.random.rand(MATRIX_SIZE, MATRIX_SIZE) 
        results[idx] = np.dot(matrix, constant_matrix)

def monitor_cpu_usage(interval, duration):
    cpu_usage = []
    for _ in range(int(duration / interval)):
        cpu_usage.append(psutil.cpu_percent(interval=interval))
    return cpu_usage

execution_times = []
cpu_usages = {}

for threads in THREAD_COUNTS:
    constant_matrix = np.random.rand(MATRIX_SIZE, MATRIX_SIZE) 
    results = [None] * NUM_MATRICES  # Placeholder for results

    start_time = time.time()
    try:
        with ThreadPoolExecutor(max_workers=threads) as executor:
            batch_indices = list(range(NUM_MATRICES))
            futures = []
            for i in range(0, len(batch_indices), BATCH_SIZE):
                chunk = batch_indices[i : i + BATCH_SIZE]
                futures.append(executor.submit(multiply_batch, chunk, constant_matrix, results))
            for future in futures:
                future.result()  # Ensure all threads complete

    except MemoryError as e:
        print(f"Error: {e}")
        break

    end_time = time.time()
    execution_times.append(end_time - start_time)

    cpu_usages[threads] = monitor_cpu_usage(interval=0.5, duration=end_time - start_time)

plt.figure(figsize=(10, 5))
plt.plot(THREAD_COUNTS, execution_times, marker='o', label="Execution Time")
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
