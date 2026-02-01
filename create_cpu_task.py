import psutil
import random
import csv
from datetime import datetime
import numpy as np

def get_cpu_speed():
    """Get CPU frequency in Hz. Fallback to 2.5GHz if detection fails."""
    try:
        freq = psutil.cpu_freq()
        if freq:
            return freq.max * 1_000_000  # Convert MHz to Hz
    except:
        pass
    return 2.5e9  # Fallback: 2.5 GHz

# Global CPU speed for calibration
CPU_CYCLES_PER_SEC = get_cpu_speed()
print(f"Detected CPU Calibration Speed: {CPU_CYCLES_PER_SEC/1e9:.2f} GHz")

def generate_task(task_id, task_type):
    """Generate a task of specific type with realistic parameters based on current hardware"""
    
    # Define deadlines first (seconds)
    # Light: fast response (100ms - 500ms)
    # Medium: interactive (500ms - 2s)
    # Heavy: batch processing (1s - 5s)
    deadline_ranges = {
        'light': (0.1, 0.5),
        'medium': (0.5, 2.0),
        'heavy': (1.0, 5.0)
    }

    deadline_range = deadline_ranges[task_type]
    deadline = random.uniform(*deadline_range)

    # Calculate max possible cycles this specific laptop can do in that deadline
    max_cycles = deadline * CPU_CYCLES_PER_SEC

    # Task Difficulty Factors (percentage of max utilization)
    # Light: Uses 1% - 10% of CPU time within deadline
    # Medium: Uses 30% - 70% of CPU time within deadline
    # Heavy: Uses 60% - 110% of CPU time (some might miss deadline on purpose!)
    difficulty_profiles = {
        'light': (0.01, 0.10),
        'medium': (0.30, 0.70),
        'heavy': (0.60, 1.10)
    }

    diff_min, diff_max = difficulty_profiles[task_type]
    utilization = random.uniform(diff_min, diff_max)
    workload = max_cycles * utilization

    # Data sizes (RAM/Network usage)
    # Light: 1KB - 100KB
    # Medium: 100KB - 5MB
    # Heavy: 5MB - 50MB
    data_size_ranges = {
        'light': (1e3, 1e5),
        'medium': (1e5, 5e6),
        'heavy': (5e6, 50e6)
    }
    
    params_data = data_size_ranges[task_type]
    data_size = random.uniform(*params_data)

    # Result ratio (output size relative to input)
    result_ratios = {
        'light': 0.1,
        'medium': 0.2,
        'heavy': 0.3
    }
    result_size = data_size * result_ratios[task_type] * random.uniform(0.8, 1.2)

    # Priority Weights
    priority_weights = {
        'light': [0.2, 0.6, 0.2],    # Mostly Medium
        'medium': [0.1, 0.5, 0.4],   # Medium/High
        'heavy': [0.05, 0.3, 0.65]   # Mostly High
    }
    
    priority = random.choices([0, 1, 2], weights=priority_weights[task_type], k=1)[0]
    
    # Output
    return {
        'task_id': task_id,
        'workload': round(workload, 2),
        'data_size': round(data_size, 2),
        'result_size': round(result_size, 2),
        'priority': priority,
        'deadline': round(deadline, 2),
        'type': task_type
    }

def generate_tasks(num_tasks=10000):
    """Generate a list of tasks with diverse characteristics"""
    tasks = []
    
    # Define task type distribution
    task_distribution = ['light'] * 4000 + ['medium'] * 4000 + ['heavy'] * 2000
    random.shuffle(task_distribution)
    
    for i in range(num_tasks):
        task_type = task_distribution[i]
        task = generate_task(i + 1, task_type)
        tasks.append(task)
        
        # Print progress
        if (i + 1) % 1000 == 0:
            print(f"Generated {i + 1}/{num_tasks} tasks...")
    
    return tasks

def save_tasks_to_csv(tasks, filename='diverse_tasks.csv'):
    """Save tasks to a CSV file"""
    if not tasks:
        print("No tasks to save!")
        return
    
    with open(filename, 'w', newline='') as f:
        fieldnames = ['task_id', 'workload', 'data_size', 'result_size', 'priority', 'deadline', 'type']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(tasks)
    
    print(f"\nSuccessfully saved {len(tasks)} tasks to {filename}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    print("Generating 10,000 diverse tasks...")
    tasks = generate_tasks(10000)
    
    # Save to CSV
    save_tasks_to_csv(tasks)
    
    # Print some statistics
    task_types = [t['type'] for t in tasks]
    print("\nTask Type Distribution:")
    print(f"Light tasks: {task_types.count('light')} (40%)")
    print(f"Medium tasks: {task_types.count('medium')} (40%)")
    print(f"Heavy tasks: {task_types.count('heavy')} (20%)")
    
    # Print first 3 tasks as example
    print("\nSample Tasks:")
    for i, task in enumerate(tasks[:3], 1):
        print(f"\nTask {i}:")
        for key, value in task.items():
            print(f"  {key}: {value}")
