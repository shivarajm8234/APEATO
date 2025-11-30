import json
import os
import random
from datetime import datetime
from APEATO import (
    APEATOAlgorithm,
    Task,
    SystemState,
    Priority,
    Activity,
    Location
)

RESULT_DIR = "results_sensitivity"
os.makedirs(RESULT_DIR, exist_ok=True)

def generate_state_with_params(params):
    """Generate system state but override key sensitivity parameters."""

    # Override bandwidth based on sensitivity
    bandwidth = params["network_delay_ms"]
    # Convert delay to approximate bandwidth effect (simple inverse proxy)
    bandwidth = max(1e6, 100e6 / (1 + bandwidth)) 

    return SystemState(
        battery=random.uniform(15, 95),
        cpu_load=random.uniform(10, 90),
        bandwidth=bandwidth,
        time_of_day=datetime.now().hour,
        activity=random.choice(list(Activity))
    )

def generate_task():
    """Generate a random realistic task."""
    return Task(
        task_id=random.randint(1, 999),
        workload=random.uniform(1e9, 12e9),
        data_size=random.uniform(1e6, 20e6),
        result_size=random.uniform(0.2e6, 3e6),
        priority=random.choice(list(Priority)),
        deadline=random.uniform(0.3, 4.0)
    )

def run_simulation(sim_index, params):
    apeato = APEATOAlgorithm()

    # Override CPU multipliers (affects f_E and f_C)
    apeato.f_E = 5e9 * params["edge_cpu_multiplier"]
    apeato.f_C = 20e9 * params["cloud_cpu_multiplier"]

    state = generate_state_with_params(params)
    task = generate_task()

    decision = apeato.make_decision(task, state)
    result = apeato.execute_and_learn(task, state, decision)

    output = {
        "simulation_index": sim_index,
        "params": params,
        "task": {
            "workload": task.workload,
            "data_size": task.data_size,
            "result_size": task.result_size,
            "priority": task.priority.name,
            "deadline": task.deadline,
        },
        "state": {
            "battery": state.battery,
            "cpu_load": state.cpu_load,
            "bandwidth": state.bandwidth,
            "time_of_day": state.time_of_day,
            "activity": state.activity.name,
        },
        "decision": decision.name,
        "energy": result["energy"],
        "latency": result["latency"],
        "reward": result["reward"],
        "optimal": result["optimal"]
    }

    # Save JSON
    out_path = os.path.join(RESULT_DIR, f"result_{sim_index}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=4)

    print(f"✅ Saved simulation result {sim_index}: {out_path}")
    return decision


def main():
    sensitivity_params = []

    cloud_cpu_vals = [0.8, 1.0]
    edge_cpu_vals = [0.5, 1.0, 2.0]
    delays = [10, 50, 100]
    seeds = [1, 2, 3]

    for c in cloud_cpu_vals:
        for e in edge_cpu_vals:
            for d in delays:
                for s in seeds:
                    sensitivity_params.append({
                        "cloud_cpu_multiplier": c,
                        "edge_cpu_multiplier": e,
                        "network_delay_ms": d,
                        "random_seed": s
                    })

    print(f"Running {len(sensitivity_params)} simulations...\n")

    decision_counts = {
        Location.DEVICE: 0,
        Location.EDGE: 0,
        Location.CLOUD: 0
    }

    for idx, params in enumerate(sensitivity_params):
        random.seed(params["random_seed"])
        print(f"--- Simulation {idx+1}/{len(sensitivity_params)} ---")
        decision = run_simulation(idx, params)
        decision_counts[decision] += 1

    print("\n==================== SUMMARY ====================")
    total = sum(decision_counts.values())
    for loc, count in decision_counts.items():
        pct = (count / total) * 100
        print(f"{loc.name}: {count} ({pct:.1f}%)")
    print("=================================================\n")


if __name__ == "__main__":
    main()
