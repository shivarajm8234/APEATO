import numpy as np
import math
from collections import deque
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from enum import Enum

class Location(Enum):
    DEVICE = 0
    EDGE = 1
    CLOUD = 2

class Priority(Enum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2

class Activity(Enum):
    IDLE = 0
    NORMAL = 1
    ACTIVE = 2
    GAMING = 3

@dataclass
class Task:
    task_id: int
    workload: float  # W_i in cycles
    data_size: float  # S_i in bits
    result_size: float  # R_i in bits
    priority: Priority
    deadline: float  # in seconds

@dataclass
class SystemState:
    battery: float  # 0-100%
    cpu_load: float  # 0-100%
    bandwidth: float  # bps
    time_of_day: int  # 0-23 hours
    activity: Activity

class APEATOAlgorithm:
    def __init__(self):
        # Device parameters
        self.P_comp_D = 10.0  # W
        self.P_idle_D = 1.0  # W
        self.P_trans = 2.0  # W
        self.f_D = 1.5e9  # Hz (1.5 GHz)
        self.f_E = 5e9  # Hz (5 GHz)
        self.f_C = 20e9  # Hz (20 GHz)
        
        # Network parameters
        self.B_E = 50e6  # bps to edge (50 Mbps)
        self.B_C = 100e6  # bps to cloud (100 Mbps)
        self.eta_C = 0.75  # network efficiency
        self.t_prop_E = 0.005  # 5ms
        self.t_prop_C = 0.050  # 50ms
        self.t_queue_C = 0.010  # 10ms
        
        # CRITICAL: Offloading cost multipliers to prevent cloud dominance
        self.edge_overhead = 1.3  # 30% overhead for edge offloading
        self.cloud_overhead = 1.8  # 80% overhead for cloud offloading
        self.network_penalty_weight = 0.3  # Weight for network costs
        
        # Weight parameters
        self.alpha_base = 0.5
        self.alpha_battery = 0.4
        self.alpha_time = 0.15
        self.alpha_activity = 0.2
        
        # Penalty coefficients
        self.lambda_high = 0.5
        self.lambda_low = 0.3
        self.mu = 0.5
        self.nu = 0.3
        
        # Learning parameters
        self.alpha_Q = 0.2
        self.gamma_Q = 0.9
        self.epsilon_max = 0.3
        self.epsilon_min = 0.05
        self.lambda_decay = 0.001
        
        # Prediction parameters
        self.alpha_bandwidth = 0.3
        self.alpha_battery_pred = 0.4
        self.alpha_cpu = 0.35
        self.beta_variance = 0.2
        
        # Reward coefficients
        self.eta_E = 0.6
        self.eta_L = 0.4
        self.gamma_reward = 10
        
        # History storage
        self.history = deque(maxlen=1000)
        self.battery_history = deque(maxlen=50)
        self.bandwidth_history = deque(maxlen=50)
        self.cpu_history = deque(maxlen=50)
        
        # Predictions
        self.battery_pred = 100.0
        self.bandwidth_pred = 50e6
        self.cpu_pred = 50.0
        self.battery_variance = 0.0
        
        # Q-Learning table
        self.Q_table = {}
        self.decision_count = 0
        
        # Network reliability
        self.R_E = 0.95
        self.R_C = 0.90
    
    def battery_influence(self, battery: float) -> float:
        """Calculate battery influence β(t)"""
        if battery >= 80:
            return 0.0
        elif battery >= 50:
            return 0.2 * (80 - battery) / 30
        elif battery >= 20:
            return 0.2 + 0.5 * (50 - battery) / 30
        else:
            return 0.7 + 0.3 * (20 - battery) / 20
    
    def time_influence(self, hour: int) -> float:
        """Calculate time-of-day influence τ(t)"""
        return 0.3 * math.sin(2 * math.pi * (hour - 6) / 24)
    
    def activity_influence(self, activity: Activity) -> float:
        """Calculate user activity influence ρ(t)"""
        influence_map = {
            Activity.IDLE: -0.2,
            Activity.NORMAL: 0.0,
            Activity.ACTIVE: 0.3,
            Activity.GAMING: 0.5
        }
        return influence_map[activity]
    
    def predict_battery(self, current_battery: float) -> float:
        """EWMA prediction for battery"""
        self.battery_history.append(current_battery)
        self.battery_pred = (self.alpha_battery_pred * current_battery + 
                            (1 - self.alpha_battery_pred) * self.battery_pred)
        
        if len(self.battery_history) > 1:
            error = current_battery - self.battery_pred
            self.battery_variance = (self.beta_variance * error**2 + 
                                    (1 - self.beta_variance) * self.battery_variance)
        
        return self.battery_pred
    
    def predict_bandwidth(self, current_bandwidth: float) -> float:
        """EWMA prediction for bandwidth"""
        self.bandwidth_history.append(current_bandwidth)
        self.bandwidth_pred = (self.alpha_bandwidth * current_bandwidth + 
                              (1 - self.alpha_bandwidth) * self.bandwidth_pred)
        return self.bandwidth_pred
    
    def predict_cpu_load(self, current_cpu: float) -> float:
        """EWMA prediction for CPU load"""
        self.cpu_history.append(current_cpu)
        self.cpu_pred = (self.alpha_cpu * current_cpu + 
                        (1 - self.alpha_cpu) * self.cpu_pred)
        return self.cpu_pred
    
    def compute_dynamic_weights(self, state: SystemState) -> Tuple[float, float]:
        """Compute w_E(t) and w_L(t)"""
        beta = self.battery_influence(state.battery)
        tau = self.time_influence(state.time_of_day)
        rho = self.activity_influence(state.activity)
        
        w_E = self.alpha_base + self.alpha_battery * beta + \
              self.alpha_time * tau + self.alpha_activity * rho
        w_E = max(0.0, min(1.0, w_E))
        w_L = 1.0 - w_E
        
        return w_E, w_L
    
    def calculate_energy_device(self, task: Task) -> float:
        """Calculate energy for local execution"""
        t_comp = task.workload / self.f_D
        E_D = self.P_comp_D * t_comp
        return E_D
    
    def calculate_energy_edge(self, task: Task) -> float:
        """Calculate energy for edge offloading - INCLUDES NETWORK COSTS"""
        # Transmission energy (upload + download)
        t_trans_up = task.data_size / self.B_E
        t_trans_down = task.result_size / self.B_E
        E_trans = self.P_trans * (t_trans_up + t_trans_down)
        
        # Idle energy during computation
        t_comp = task.workload / self.f_E
        E_idle = self.P_idle_D * (t_comp + self.t_prop_E * 2)  # Round trip
        
        # Apply overhead multiplier to account for offloading costs
        total_energy = (E_trans + E_idle) * self.edge_overhead
        
        return total_energy
    
    def calculate_energy_cloud(self, task: Task) -> float:
        """Calculate energy for cloud offloading - INCLUDES NETWORK COSTS"""
        # Transmission energy (upload + download with efficiency factor)
        t_trans_up = task.data_size / (self.B_C * self.eta_C)
        t_trans_down = task.result_size / (self.B_C * self.eta_C)
        E_trans = self.P_trans * (t_trans_up + t_trans_down)
        
        # Idle energy during computation and queue
        t_comp = task.workload / self.f_C
        E_idle = self.P_idle_D * (t_comp + self.t_prop_C * 2 + self.t_queue_C)
        
        # Apply overhead multiplier to account for offloading costs
        total_energy = (E_trans + E_idle) * self.cloud_overhead
        
        return total_energy
    
    def calculate_latency_device(self, task: Task) -> float:
        """Calculate latency for local execution"""
        return task.workload / self.f_D
    
    def calculate_latency_edge(self, task: Task) -> float:
        """Calculate latency for edge offloading - FULL ROUND TRIP"""
        t_trans_up = task.data_size / self.B_E
        t_trans_down = task.result_size / self.B_E
        t_comp = task.workload / self.f_E
        
        # Total latency includes full round trip
        total_latency = t_trans_up + self.t_prop_E + t_comp + self.t_prop_E + t_trans_down
        
        return total_latency
    
    def calculate_latency_cloud(self, task: Task) -> float:
        """Calculate latency for cloud offloading - FULL ROUND TRIP"""
        t_trans_up = task.data_size / (self.B_C * self.eta_C)
        t_trans_down = task.result_size / (self.B_C * self.eta_C)
        t_comp = task.workload / self.f_C
        
        # Total latency includes full round trip + queue delay
        total_latency = (t_trans_up + self.t_prop_C + self.t_queue_C + 
                        t_comp + self.t_prop_C + t_trans_down)
        
        return total_latency
    
    def compute_penalties(self, task: Task, location: Location, 
                         state: SystemState, energies: Dict, latencies: Dict) -> float:
        """Compute combined penalty function"""
        penalty = 0.0
        
        # Battery penalty (only for device execution)
        if location == Location.DEVICE:
            if state.battery < 30:
                if state.battery >= 10:
                    penalty += 3 * (30 - state.battery) / 30
                else:
                    penalty += 3 + 7 * (10 - state.battery) / 10
        
        # Priority penalty
        E_max = max(energies.values())
        L_max = max(latencies.values())
        L_norm = latencies[location] / L_max if L_max > 0 else 0
        E_norm = energies[location] / E_max if E_max > 0 else 0
        
        if task.priority == Priority.HIGH:
            penalty += self.lambda_high * L_norm
        elif task.priority == Priority.LOW:
            penalty -= self.lambda_low * E_norm
        
        # Network reliability penalty (for offloading)
        if location == Location.EDGE:
            penalty += self.mu * (1 - self.R_E) * self.network_penalty_weight
        elif location == Location.CLOUD:
            penalty += self.nu * (1 - self.R_C) * self.network_penalty_weight
        
        return penalty
    
    def make_decision(self, task: Task, state: SystemState) -> Location:
        """
        MAIN DECISION FUNCTION - FIXED VERSION
        Now properly calculates costs and picks best location
        """
        # Step 1: Predict future system state
        battery_pred = self.predict_battery(state.battery)
        bandwidth_pred = self.predict_bandwidth(state.bandwidth)
        cpu_pred = self.predict_cpu_load(state.cpu_load)
        
        # Step 2: Compute dynamic weights
        w_E, w_L = self.compute_dynamic_weights(state)
        
        # Step 3: Calculate ACTUAL costs for each location
        energies = {
            Location.DEVICE: self.calculate_energy_device(task),
            Location.EDGE: self.calculate_energy_edge(task),
            Location.CLOUD: self.calculate_energy_cloud(task)
        }
        
        latencies = {
            Location.DEVICE: self.calculate_latency_device(task),
            Location.EDGE: self.calculate_latency_edge(task),
            Location.CLOUD: self.calculate_latency_cloud(task)
        }
        
        # Step 4: Normalize values
        E_max = max(energies.values())
        L_max = max(latencies.values())
        
        E_min = min(energies.values())
        L_min = min(latencies.values())
        
        # Use min-max normalization to prevent any location from dominating
        costs = {}
        for loc in Location:
            # Normalize to [0, 1] range
            E_norm = (energies[loc] - E_min) / (E_max - E_min) if E_max > E_min else 0
            L_norm = (latencies[loc] - L_min) / (L_max - L_min) if L_max > L_min else 0
            
            # Calculate penalty
            penalty = self.compute_penalties(task, loc, state, energies, latencies)
            
            # Combined cost with weights
            costs[loc] = w_E * E_norm + w_L * L_norm + penalty
        
        # Step 5: Select optimal location (minimum cost)
        optimal_location = min(costs, key=costs.get)
        
        # Step 6: Deadline check
        if latencies[optimal_location] > task.deadline:
            # Find feasible locations that meet deadline
            feasible = [loc for loc in Location 
                       if latencies[loc] <= task.deadline]
            
            if feasible:
                # Among feasible, choose minimum energy
                optimal_location = min(feasible, key=lambda l: energies[l])
            else:
                # No feasible solution, choose fastest
                optimal_location = min(latencies, key=latencies.get)
        
        # Update decision count
        self.decision_count += 1
        
        # Debug info (optional)
        if self.decision_count % 10 == 0:
            print(f"\n--- Decision #{self.decision_count} Debug ---")
            print(f"Battery: {state.battery:.1f}%, Weights: E={w_E:.2f}, L={w_L:.2f}")
            print(f"Energies: D={energies[Location.DEVICE]:.2f}, E={energies[Location.EDGE]:.2f}, C={energies[Location.CLOUD]:.2f}")
            print(f"Latencies: D={latencies[Location.DEVICE]:.3f}, E={latencies[Location.EDGE]:.3f}, C={latencies[Location.CLOUD]:.3f}")
            print(f"Costs: D={costs[Location.DEVICE]:.3f}, E={costs[Location.EDGE]:.3f}, C={costs[Location.CLOUD]:.3f}")
            print(f"Decision: {optimal_location.name}")
        
        return optimal_location
    
    def discretize_state(self, state: SystemState) -> Tuple:
        """Discretize state for Q-learning"""
        battery_bin = 0 if state.battery < 33 else (1 if state.battery < 67 else 2)
        cpu_bin = 0 if state.cpu_load < 33 else (1 if state.cpu_load < 67 else 2)
        bw_bin = 0 if state.bandwidth < 30e6 else (1 if state.bandwidth < 70e6 else 2)
        
        if state.time_of_day < 6 or state.time_of_day >= 22:
            time_bin = 3
        elif state.time_of_day < 12:
            time_bin = 0
        elif state.time_of_day < 18:
            time_bin = 1
        else:
            time_bin = 2
        
        return (battery_bin, cpu_bin, bw_bin, time_bin, state.activity.value)
    
    def discretize_action(self, w_E: float) -> Tuple[float, float]:
        """Discretize action space"""
        actions = [(0.8, 0.2), (0.7, 0.3), (0.6, 0.4), (0.5, 0.5),
                   (0.4, 0.6), (0.3, 0.7), (0.2, 0.8)]
        
        closest = min(actions, key=lambda a: abs(a[0] - w_E))
        return closest
    
    def get_epsilon(self) -> float:
        """Calculate exploration rate"""
        return self.epsilon_min + (self.epsilon_max - self.epsilon_min) * \
               math.exp(-self.lambda_decay * self.decision_count)
    
    def select_action(self, state_tuple: Tuple) -> Tuple[float, float]:
        """ε-greedy action selection"""
        epsilon = self.get_epsilon()
        
        if np.random.random() < epsilon:
            actions = [(0.8, 0.2), (0.7, 0.3), (0.6, 0.4), (0.5, 0.5),
                      (0.4, 0.6), (0.3, 0.7), (0.2, 0.8)]
            return actions[np.random.randint(len(actions))]
        else:
            if state_tuple not in self.Q_table:
                return (0.5, 0.5)
            
            best_action = max(self.Q_table[state_tuple].items(), 
                            key=lambda x: x[1])[0]
            return best_action
    
    def update_q_value(self, state: Tuple, action: Tuple[float, float], 
                      reward: float, next_state: Tuple):
        """Update Q-table using Q-learning"""
        if state not in self.Q_table:
            self.Q_table[state] = {}
        if action not in self.Q_table[state]:
            self.Q_table[state][action] = 0.0
        
        max_next_q = 0.0
        if next_state in self.Q_table and self.Q_table[next_state]:
            max_next_q = max(self.Q_table[next_state].values())
        
        current_q = self.Q_table[state][action]
        self.Q_table[state][action] = current_q + self.alpha_Q * \
            (reward + self.gamma_Q * max_next_q - current_q)
    
    def compute_reward(self, E_actual: float, L_actual: float,
                      E_optimal: float, L_optimal: float) -> float:
        """Compute reward for learning"""
        is_optimal = (abs(E_actual - E_optimal) < 0.1 * E_optimal and
                     abs(L_actual - L_optimal) < 0.1 * L_optimal)
        
        reward = -self.eta_E * E_actual - self.eta_L * L_actual
        if is_optimal:
            reward += self.gamma_reward
        
        return reward
    
    def execute_and_learn(self, task: Task, state: SystemState, 
                         location: Location) -> Dict:
        """Execute task and update learning models"""
        # Simulate execution
        if location == Location.DEVICE:
            E_actual = self.calculate_energy_device(task)
            L_actual = self.calculate_latency_device(task)
        elif location == Location.EDGE:
            E_actual = self.calculate_energy_edge(task)
            L_actual = self.calculate_latency_edge(task)
        else:
            E_actual = self.calculate_energy_cloud(task)
            L_actual = self.calculate_latency_cloud(task)
        
        # Calculate optimal values
        E_optimal = min(self.calculate_energy_device(task),
                       self.calculate_energy_edge(task),
                       self.calculate_energy_cloud(task))
        L_optimal = min(self.calculate_latency_device(task),
                       self.calculate_latency_edge(task),
                       self.calculate_latency_cloud(task))
        
        # Compute reward
        reward = self.compute_reward(E_actual, L_actual, E_optimal, L_optimal)
        
        # Update Q-learning
        current_state = self.discretize_state(state)
        w_E, w_L = self.compute_dynamic_weights(state)
        action = self.discretize_action(w_E)
        
        self.update_q_value(current_state, action, reward, current_state)
        
        # Store history
        self.history.append({
            'task': task,
            'location': location,
            'energy': E_actual,
            'latency': L_actual,
            'state': state
        })
        
        return {
            'location': location,
            'energy': E_actual,
            'latency': L_actual,
            'reward': reward,
            'optimal': abs(E_actual - E_optimal) < 0.1 * E_optimal
        }

# Example usage
if __name__ == "__main__":
    apeato = APEATOAlgorithm()
    
    print("="*70)
    print("APEATO Algorithm - Fixed Core Decision Logic")
    print("="*70)
    
    # Test with different scenarios
    scenarios = [
        ("High Battery, Light Task", 90, 2000, 5e6, 1e6),
        ("Medium Battery, Medium Task", 50, 5000, 10e6, 2e6),
        ("Low Battery, Heavy Task", 20, 8000, 15e6, 3e6),
        ("Critical Battery, Light Task", 10, 3000, 8e6, 1e6),
    ]
    
    for scenario_name, battery, cycles, data_size, result_size in scenarios:
        print(f"\n{'='*70}")
        print(f"Scenario: {scenario_name}")
        print(f"{'='*70}")
        
        task = Task(
            task_id=1,
            workload=cycles * 1e6,
            data_size=data_size * 8,
            result_size=result_size * 8,
            priority=Priority.MEDIUM,
            deadline=2.0
        )
        
        state = SystemState(
            battery=battery,
            cpu_load=50.0,
            bandwidth=55e6,
            time_of_day=14,
            activity=Activity.NORMAL
        )
        
        decision = apeato.make_decision(task, state)
        result = apeato.execute_and_learn(task, state, decision)
        
        print(f"\nFinal Decision: {decision.name}")
        print(f"Energy: {result['energy']:.4f} J")
        print(f"Latency: {result['latency']:.4f} s")
        print(f"Optimal: {result['optimal']}")