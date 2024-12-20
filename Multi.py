import numpy as np
import time
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import psutil

# Constants
MATRIX_SIZE = 5000  # Size of each matrix
NUM_MATRICES = 500  # Total number of matrices
THREAD_COUNTS = [1, 2, 3, 4, 5, 6, 7, 8]  # Logical threads for i5-9300H
BATCH_SIZE = 2  # Process only 2 matrices at a time to fit within memory constraints

# Function to multiply a batch of matrices
def multiply_batch(batch_indices, constant_matrix, results):
    for idx in batch_indices:
        matrix = np.random.rand(MATRIX_SIZE, MATRIX_SIZE)  # Generate on-the-fly
        results[idx] = np.dot(matrix, constant_matrix)

# Function to monitor CPU usage during execution
def monitor_cpu_usage(interval, duration):
    cpu_usage = []
    for _ in range(int(duration / interval)):
        cpu_usage.append(psutil.cpu_percent(interval=interval))
    return cpu_usage

# Main execution
execution_times = []
cpu_usages = {}

for threads in THREAD_COUNTS:
    constant_matrix = np.random.rand(MATRIX_SIZE, MATRIX_SIZE)  # Shared constant matrix
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

    # Monitor CPU usage for this thread count
    cpu_usages[threads] = monitor_cpu_usage(interval=0.5, duration=end_time - start_time)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(THREAD_COUNTS, execution_times, marker='o', label="Execution Time")
plt.title('Execution Time vs Number of Threads (8GB RAM)')
plt.xlabel('Number of Threads')
plt.ylabel('Execution Time (seconds)')
plt.legend()
plt.grid()
plt.show()

# Print CPU usage for each thread count
for threads, usage in cpu_usages.items():
    print(f"Threads: {threads}, Avg CPU Usage: {np.mean(usage):.2f}%")

# Generate a CPU usage plot for visualization
plt.figure(figsize=(10, 5))
for threads, usage in cpu_usages.items():
    plt.plot(range(len(usage)), usage, label=f"{threads} Threads")
plt.title("CPU Usage vs Time")
plt.xlabel("Time (steps)")
plt.ylabel("CPU Usage (%)")
plt.legend()
plt.grid()
plt.show()
