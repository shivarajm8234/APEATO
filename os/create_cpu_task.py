import psutil
import random
import csv
from datetime import datetime
import numpy as np

def generate_task(task_id, task_type):
    """Generate a task of specific type with realistic parameters"""
    # Base parameters for different task types
    task_types = {
        'light': {
            'workload_range': (1e6, 1e8),      # 1M to 100M cycles
            'data_size_range': (1e3, 1e5),     # 1KB to 100KB
            'result_ratio': 0.1,               # 10% of data size
            'deadline_range': (0.1, 0.5),      # 0.1 to 0.5 seconds
            'priority_weights': [0.2, 0.6, 0.2]  # 20% low, 60% medium, 20% high
        },
        'medium': {
            'workload_range': (1e8, 5e9),      # 100M to 5B cycles
            'data_size_range': (1e5, 5e6),     # 100KB to 5MB
            'result_ratio': 0.2,               # 20% of data size
            'deadline_range': (0.5, 2.0),      # 0.5 to 2 seconds
            'priority_weights': [0.1, 0.5, 0.4]  # 10% low, 50% medium, 40% high
        },
        'heavy': {
            'workload_range': (5e9, 20e9),     # 5B to 20B cycles
            'data_size_range': (5e6, 50e6),    # 5MB to 50MB
            'result_ratio': 0.3,               # 30% of data size
            'deadline_range': (1.0, 5.0),      # 1 to 5 seconds
            'priority_weights': [0.05, 0.3, 0.65]  # 5% low, 30% medium, 65% high
        }
    }
    
    params = task_types[task_type]
    
    # Generate task parameters
    workload = random.uniform(*params['workload_range'])
    data_size = random.uniform(*params['data_size_range'])
    result_size = data_size * params['result_ratio'] * random.uniform(0.8, 1.2)
    priority = random.choices([0, 1, 2], weights=params['priority_weights'], k=1)[0]
    deadline = random.uniform(*params['deadline_range'])
    
    # Add some noise to make it more realistic
    workload *= random.uniform(0.9, 1.1)
    data_size = max(1, data_size * random.uniform(0.9, 1.1))
    result_size = max(1, result_size * random.uniform(0.9, 1.1))
    
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