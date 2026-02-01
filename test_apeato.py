#!/usr/bin/env python3
import csv
import random
from datetime import datetime
from APEATO import APEATOAlgorithm, Task, SystemState, Priority, Activity, Location
# NOTE: We import Location FROM APEATO to ensure the enums match.

def read_tasks_from_csv(file_path):
    """Read tasks from CSV file and return a list of Task objects"""
    tasks = []
    priority_map = {'0': Priority.LOW, '1': Priority.MEDIUM, '2': Priority.HIGH}
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # defensive casting — if workload is empty or 0 use a small default to avoid zeros
            workload = float(row.get('workload') or 0.0)
            data_size = float(row.get('data_size') or 0.0)
            result_size = float(row.get('result_size') or 0.0)
            task = Task(
                task_id=int(row.get('task_id', 0)),
                workload=workload,
                data_size=data_size,
                result_size=result_size,
                priority=priority_map[row['priority']],
                deadline=float(row.get('deadline', 1.0))
            )
            tasks.append(task)
    return tasks

def generate_balanced_state():
    """Generate a system state that encourages different offloading decisions"""
    activities = list(Activity)
    current_hour = datetime.now().hour

    battery = random.choices(
        [random.uniform(10, 30), random.uniform(30, 70), random.uniform(70, 100)],
        weights=[0.4, 0.3, 0.3],
        k=1
    )[0]

    if 8 <= current_hour <= 18:
        cpu_load = random.uniform(40, 90)
    else:
        cpu_load = random.uniform(10, 60)

    bandwidth = random.choices(
        [random.uniform(1e6, 10e6), random.uniform(10e6, 50e6), random.uniform(50e6, 100e6)],
        weights=[0.3, 0.4, 0.3],
        k=1
    )[0]

    if 8 <= current_hour <= 20:
        activity_weights = [0.1, 0.4, 0.3, 0.2]
    else:
        activity_weights = [0.3, 0.5, 0.1, 0.1]

    activity = random.choices(activities, weights=activity_weights, k=1)[0]

    return SystemState(
        battery=battery,
        cpu_load=cpu_load,
        bandwidth=bandwidth,
        time_of_day=current_hour,
        activity=activity
    )

def get_decision_emoji_by_name(decision_name: str):
    """Get an emoji for the decision type using the decision name string"""
    emoji_map = {
        'DEVICE': "📱 DEVICE",
        'EDGE': "🖥️ EDGE",
        'CLOUD': "☁️ CLOUD"
    }
    return emoji_map.get(decision_name, decision_name)

def print_decision(task, state, decision, result):
    """Print the decision and results in a formatted way (safe and robust)"""
    COLORS = {
        'HEADER': '\033[95m', 'BLUE': '\033[94m', 'CYAN': '\033[96m',
        'GREEN': '\033[92m', 'YELLOW': '\033[93m', 'RED': '\033[91m',
        'ENDC': '\033[0m', 'BOLD': '\033[1m'
    }

    # resolve decision name safely
    if hasattr(decision, 'name'):
        decision_name = decision.name
    else:
        decision_name = str(decision)

    decision_color = {
        'DEVICE': COLORS['GREEN'],
        'EDGE': COLORS['BLUE'],
        'CLOUD': COLORS['YELLOW']
    }.get(decision_name, COLORS['CYAN'])

    print("\n" + "="*100)
    print(f"{COLORS['BOLD']}TASK {task.task_id} DETAILS:{COLORS['ENDC']}")
    print(f"- Workload: {COLORS['CYAN']}{task.workload/1e9:.2f} billion cycles{COLORS['ENDC']}")
    print(f"- Data Size: {COLORS['CYAN']}{task.data_size/1e6:.2f} MB{COLORS['ENDC']}")
    print(f"- Result Size: {COLORS['CYAN']}{task.result_size/1e6:.2f} MB{COLORS['ENDC']}")
    print(f"- Priority: {COLORS['CYAN']}{task.priority.name}{COLORS['ENDC']}")
    print(f"- Deadline: {COLORS['CYAN']}{task.deadline:.2f} seconds{COLORS['ENDC']}")

    print(f"\n{COLORS['BOLD']}SYSTEM STATE:{COLORS['ENDC']}")
    battery_color = COLORS['GREEN']
    if state.battery < 30:
        battery_color = COLORS['RED']
    elif state.battery < 60:
        battery_color = COLORS['YELLOW']
    print(f"- Battery: {battery_color}{state.battery:.1f}%{COLORS['ENDC']}")

    cpu_color = COLORS['GREEN']
    if state.cpu_load > 70:
        cpu_color = COLORS['RED']
    elif state.cpu_load > 40:
        cpu_color = COLORS['YELLOW']
    print(f"- CPU Load: {cpu_color}{state.cpu_load:.1f}%{COLORS['ENDC']}")

    bw_color = COLORS['GREEN']
    if state.bandwidth < 10e6:
        bw_color = COLORS['RED']
    elif state.bandwidth < 30e6:
        bw_color = COLORS['YELLOW']
    print(f"- Bandwidth: {bw_color}{state.bandwidth/1e6:.1f} Mbps{COLORS['ENDC']}")
    print(f"- Time of Day: {COLORS['CYAN']}{state.time_of_day}:00{COLORS['ENDC']}")
    print(f"- Activity: {COLORS['CYAN']}{state.activity.name}{COLORS['ENDC']}")

    print(f"\n{COLORS['BOLD']}DECISION:{COLORS['ENDC']}")
    print(f"- Execution Location: {decision_color}{get_decision_emoji_by_name(decision_name)}{COLORS['ENDC']}")
    print(f"- Energy Consumed: {COLORS['CYAN']}{result['energy']:.4f} Joules{COLORS['ENDC']}")
    print(f"- Latency: {COLORS['CYAN']}{result['latency']*1000:.2f} ms{COLORS['ENDC']}")
    print(f"- Reward: {COLORS['CYAN']}{result['reward']:.4f}{COLORS['ENDC']}")
    print(f"- Optimal Decision: {COLORS['GREEN'] if result['optimal'] else COLORS['YELLOW']}{'Yes' if result['optimal'] else 'No'}{COLORS['ENDC']}")
    print("="*100 + "\n")

def run_test():
    apeato = APEATOAlgorithm()
    tasks = read_tasks_from_csv('diverse_tasks.csv')

    print("\n" + "#"*50)
    print(f"{' APEATO TASK OFFLOADING SIMULATOR ':#^50}")
    print("#"*50)

    # We'll count decisions by name (strings) to be robust
    decision_counts = {'DEVICE': 0, 'EDGE': 0, 'CLOUD': 0, 'UNKNOWN': 0}

    for task in tasks:
        state = generate_balanced_state()
        decision, _ = apeato.make_decision(task, state)
        result = apeato.execute_and_learn(task, state, decision)

        # Get decision name robustly
        if hasattr(decision, 'name'):
            dname = decision.name
        else:
            dname = str(decision)

        if dname in decision_counts:
            decision_counts[dname] += 1
        else:
            decision_counts['UNKNOWN'] += 1
            print(f"Warning: Decision {decision} produced unknown name '{dname}'")

        print_decision(task, state, decision, result)

    # Summary
    total = len(tasks)
    print("\n" + "#"*50)
    print(f"{ ' DECISION SUMMARY ':#^50}")
    print(f"Total Tasks: {total}")
    for name in ['DEVICE', 'EDGE', 'CLOUD', 'UNKNOWN']:
        count = decision_counts.get(name, 0)
        pct = (count / total) * 100 if total > 0 else 0.0
        emoji = get_decision_emoji_by_name(name)
        print(f"{emoji}: {count} ({pct:.1f}%)")
    print("#"*50 + "\n")

if __name__ == "__main__":
    print("Starting APEATO Algorithm Test...\n")
    run_test()
    print("\nTest completed!")
