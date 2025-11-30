import csv
import random
import numpy as np
from datetime import datetime

def generate_realistic_task(task_id):
    # Task types with realistic characteristics
    task_types = [
        # (workload_range, data_size_range, result_ratio, deadline_range, priority_weights)
        # Light tasks (sensor data, simple processing)
        ((1e6, 1e8), (1e3, 1e5), 0.1, (0.1, 0.5), [0.2, 0.6, 0.2]),  # 60% medium priority
        # Medium tasks (image processing, medium computations)
        ((1e8, 5e9), (1e5, 5e6), 0.2, (0.5, 2.0), [0.1, 0.5, 0.4]),  # 50% medium, 40% high
        # Heavy tasks (video processing, complex ML)
        ((5e9, 20e9), (5e6, 50e6), 0.3, (1.0, 5.0), [0.05, 0.3, 0.65])  # 65% high priority
    ]
    
    # Choose task type
    task_type = random.choices(task_types, weights=[0.5, 0.35, 0.15], k=1)[0]
    
    # Generate task parameters
    workload = random.uniform(*task_type[0])
    data_size = random.uniform(*task_type[1])
    result_size = data_size * task_type[2] * random.uniform(0.8, 1.2)  # 80-120% of typical ratio
    deadline = random.uniform(*task_type[3])
    
    # Add some randomness to deadlines based on workload
    deadline = max(0.1, deadline * random.uniform(0.8, 1.5))
    
    # Choose priority (0=low, 1=medium, 2=high)
    priority = random.choices([0, 1, 2], weights=task_type[4], k=1)[0]
    
    return {
        'task_id': task_id,
        'workload': f"{workload:.2f}",
        'data_size': f"{data_size:.2f}",
        'result_size': f"{result_size:.2f}",
        'priority': str(priority),
        'deadline': f"{deadline:.2f}"
    }

def generate_tasks_csv(filename, num_tasks=1000):
    # CSV header
    fieldnames = ['task_id', 'workload', 'data_size', 'result_size', 'priority', 'deadline']
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i in range(1, num_tasks + 1):
            task = generate_realistic_task(i)
            writer.writerow(task)
            
            # Print progress
            if i % 100 == 0:
                print(f"Generated {i}/{num_tasks} tasks...")
    
    print(f"\nSuccessfully generated {num_tasks} tasks in {filename}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Generate tasks
    csv_filename = "tasks.csv"
    num_tasks = 1000
    
    print(f"Generating {num_tasks} realistic tasks...")
    generate_tasks_csv(csv_filename, num_tasks)
    
    print("\nTask generation complete!")