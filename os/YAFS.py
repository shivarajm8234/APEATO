"""
YAFS (Yet Another Fog Simulator) - Complete Simulation Environment
Integrated with APEATO Algorithm for Task Offloading
"""

import time
import random
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import deque
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class NodeType(Enum):
    DEVICE = "device"
    EDGE = "edge"
    CLOUD = "cloud"

class TaskStatus(Enum):
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class TaskDefinition:
    """Task definition for simulation"""
    task_id: int
    input_size: float  # MB
    output_size: float  # MB
    cpu_cycles: float  # Million cycles
    deadline: float  # seconds
    priority: str = "medium"  # low, medium, high
    arrival_time: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'task_id': self.task_id,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'cpu_cycles': self.cpu_cycles,
            'deadline': self.deadline,
            'priority': self.priority,
            'arrival_time': self.arrival_time
        }

@dataclass
class TaskResult:
    """Result of task execution"""
    task_id: int
    location: NodeType
    status: TaskStatus
    energy_consumed: float  # Joules
    latency: float  # seconds
    start_time: float
    end_time: float
    deadline_met: bool
    network_delay: float = 0.0
    computation_time: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'task_id': self.task_id,
            'location': self.location.value,
            'status': self.status.value,
            'energy_consumed': self.energy_consumed,
            'latency': self.latency,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'deadline_met': self.deadline_met,
            'network_delay': self.network_delay,
            'computation_time': self.computation_time
        }

# ============================================================================
# NODE CLASSES
# ============================================================================

class ComputeNode:
    """Base class for compute nodes"""
    
    def __init__(self, node_id: str, node_type: NodeType, 
                 cpu_frequency: float, power_idle: float, 
                 power_active: float):
        self.node_id = node_id
        self.node_type = node_type
        self.cpu_frequency = cpu_frequency  # MHz
        self.power_idle = power_idle  # Watts
        self.power_active = power_active  # Watts
        self.current_load = 0.0  # 0-100%
        self.task_queue = deque()
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{node_id}")
        
        # Statistics
        self.tasks_executed = 0
        self.total_energy = 0.0
        self.total_computation_time = 0.0
    
    def calculate_execution_time(self, cpu_cycles: float) -> float:
        """Calculate task execution time based on CPU cycles"""
        # cpu_cycles in millions, cpu_frequency in MHz
        exec_time = cpu_cycles / self.cpu_frequency
        # Add load-based delay
        load_factor = 1 + (self.current_load / 100) * 0.5
        return exec_time * load_factor
    
    def calculate_energy(self, execution_time: float, is_active: bool = True) -> float:
        """Calculate energy consumption"""
        if is_active:
            return self.power_active * execution_time
        return self.power_idle * execution_time
    
    def update_load(self, delta: float):
        """Update current load"""
        self.current_load = max(0, min(100, self.current_load + delta))
    
    def get_status(self) -> Dict:
        """Get node status"""
        return {
            'node_id': self.node_id,
            'type': self.node_type.value,
            'load': self.current_load,
            'tasks_in_queue': len(self.task_queue),
            'tasks_executed': self.tasks_executed,
            'total_energy': self.total_energy
        }

class Device(ComputeNode):
    """Mobile device node with battery"""
    
    def __init__(self, device_id: str = "device_0"):
        super().__init__(
            node_id=device_id,
            node_type=NodeType.DEVICE,
            cpu_frequency=1500,  # 1.5 GHz
            power_idle=1.0,  # 1W idle
            power_active=10.0  # 10W active
        )
        
        # Battery specific
        self.battery_capacity = 15000  # mAh (e.g., 15Wh at 3.7V)
        self.battery_level = 100.0  # percentage
        self.initial_battery = 100.0
        
        # Network interface
        self.tx_power = 2.0  # Watts for transmission
    
    def consume_battery(self, energy_joules: float):
        """Consume battery based on energy usage"""
        # Convert Joules to Wh: 1 Wh = 3600 J
        energy_wh = energy_joules / 3600
        # Calculate percentage of total capacity
        battery_consumed = (energy_wh / (self.battery_capacity / 1000)) * 100
        self.battery_level = max(0, self.battery_level - battery_consumed)
        self.logger.debug(f"Battery consumed: {battery_consumed:.4f}%, "
                         f"remaining: {self.battery_level:.2f}%")
    
    def execute_task(self, task: TaskDefinition, sim_time: float) -> TaskResult:
        """Execute task locally on device"""
        self.logger.info(f"Executing task {task.task_id} locally")
        
        start_time = sim_time
        
        # Calculate execution time
        exec_time = self.calculate_execution_time(task.cpu_cycles)
        
        # Calculate energy
        energy = self.calculate_energy(exec_time, is_active=True)
        
        # Update statistics
        self.consume_battery(energy)
        self.total_energy += energy
        self.total_computation_time += exec_time
        self.tasks_executed += 1
        
        # Simulate load increase during execution
        self.update_load(20)
        
        end_time = start_time + exec_time
        deadline_met = exec_time <= task.deadline
        
        # Restore load after execution
        self.update_load(-20)
        
        return TaskResult(
            task_id=task.task_id,
            location=NodeType.DEVICE,
            status=TaskStatus.COMPLETED,
            energy_consumed=energy,
            latency=exec_time,
            start_time=start_time,
            end_time=end_time,
            deadline_met=deadline_met,
            computation_time=exec_time
        )
    
    def get_status(self) -> Dict:
        """Get device status including battery"""
        status = super().get_status()
        status.update({
            'battery_level': self.battery_level,
            'battery_consumed': self.initial_battery - self.battery_level
        })
        return status

class EdgeNode(ComputeNode):
    """Edge server node"""
    
    def __init__(self, edge_id: str = "edge_0"):
        super().__init__(
            node_id=edge_id,
            node_type=NodeType.EDGE,
            cpu_frequency=5000,  # 5 GHz
            power_idle=20.0,  # 20W idle
            power_active=50.0  # 50W active
        )
        
        # Edge specific parameters
        self.max_concurrent_tasks = 10
        self.bandwidth = 50  # Mbps to/from device
    
    def execute_task(self, task: TaskDefinition, sim_time: float) -> TaskResult:
        """Execute task on edge node"""
        self.logger.info(f"Executing task {task.task_id} on edge")
        
        start_time = sim_time
        
        # Calculate execution time
        exec_time = self.calculate_execution_time(task.cpu_cycles)
        
        # Calculate energy (edge doesn't affect device battery directly)
        energy = self.calculate_energy(exec_time, is_active=True)
        
        # Update statistics
        self.total_energy += energy
        self.total_computation_time += exec_time
        self.tasks_executed += 1
        
        # Simulate load
        self.update_load(10)
        
        end_time = start_time + exec_time
        
        self.update_load(-10)
        
        return TaskResult(
            task_id=task.task_id,
            location=NodeType.EDGE,
            status=TaskStatus.COMPLETED,
            energy_consumed=0,  # Edge energy not counted toward device
            latency=exec_time,
            start_time=start_time,
            end_time=end_time,
            deadline_met=True,
            computation_time=exec_time
        )

class CloudNode(ComputeNode):
    """Cloud server node"""
    
    def __init__(self, cloud_id: str = "cloud_0"):
        super().__init__(
            node_id=cloud_id,
            node_type=NodeType.CLOUD,
            cpu_frequency=20000,  # 20 GHz (represents powerful server)
            power_idle=100.0,  # 100W idle
            power_active=500.0  # 500W active
        )
        
        # Cloud specific parameters
        self.max_concurrent_tasks = 100
        self.bandwidth = 100  # Mbps to/from device
        self.queue_delay = 0.01  # 10ms base queue delay
    
    def execute_task(self, task: TaskDefinition, sim_time: float) -> TaskResult:
        """Execute task on cloud node"""
        self.logger.info(f"Executing task {task.task_id} on cloud")
        
        start_time = sim_time
        
        # Calculate execution time
        exec_time = self.calculate_execution_time(task.cpu_cycles)
        
        # Add queue delay based on load
        queue_time = self.queue_delay * (1 + self.current_load / 100)
        total_time = exec_time + queue_time
        
        # Calculate energy
        energy = self.calculate_energy(total_time, is_active=True)
        
        # Update statistics
        self.total_energy += energy
        self.total_computation_time += exec_time
        self.tasks_executed += 1
        
        # Simulate load
        self.update_load(5)
        
        end_time = start_time + total_time
        
        self.update_load(-5)
        
        return TaskResult(
            task_id=task.task_id,
            location=NodeType.CLOUD,
            status=TaskStatus.COMPLETED,
            energy_consumed=0,  # Cloud energy not counted toward device
            latency=total_time,
            start_time=start_time,
            end_time=end_time,
            deadline_met=True,
            computation_time=exec_time
        )

# ============================================================================
# NETWORK SIMULATOR
# ============================================================================

class NetworkSimulator:
    """Simulates network communication"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Network parameters
        self.edge_latency_base = 0.005  # 5ms base
        self.cloud_latency_base = 0.050  # 50ms base
        self.edge_bandwidth = 50  # Mbps
        self.cloud_bandwidth = 100  # Mbps
        self.cloud_efficiency = 0.75  # Network efficiency factor
        
        # Device transmission power
        self.tx_power = 2.0  # Watts
        
        # Statistics
        self.total_data_transmitted = 0.0  # MB
        self.total_transmission_time = 0.0  # seconds
        self.total_transmission_energy = 0.0  # Joules
    
    def calculate_transmission(self, data_size_mb: float, bandwidth_mbps: float,
                              base_latency: float, efficiency: float = 1.0) -> Tuple[float, float]:
        """Calculate transmission time and energy"""
        # Convert MB to Mb
        data_size_mb_bits = data_size_mb * 8
        
        # Transmission time
        tx_time = (data_size_mb_bits / (bandwidth_mbps * efficiency)) + base_latency
        
        # Add random jitter (±10%)
        jitter = random.uniform(-0.1, 0.1)
        tx_time *= (1 + jitter)
        
        # Transmission energy (device only)
        tx_energy = self.tx_power * tx_time
        
        self.total_data_transmitted += data_size_mb
        self.total_transmission_time += tx_time
        self.total_transmission_energy += tx_energy
        
        return tx_time, tx_energy
    
    def transmit_to_edge(self, task: TaskDefinition) -> Tuple[float, float, float]:
        """Transmit task to edge and receive result"""
        # Upload data
        upload_time, upload_energy = self.calculate_transmission(
            task.input_size, self.edge_bandwidth, self.edge_latency_base
        )
        
        # Download result
        download_time, download_energy = self.calculate_transmission(
            task.output_size, self.edge_bandwidth, self.edge_latency_base
        )
        
        total_time = upload_time + download_time
        total_energy = upload_energy + download_energy
        
        self.logger.debug(f"Edge transmission - Time: {total_time:.4f}s, "
                         f"Energy: {total_energy:.4f}J")
        
        return total_time, total_energy, self.edge_latency_base
    
    def transmit_to_cloud(self, task: TaskDefinition) -> Tuple[float, float, float]:
        """Transmit task to cloud and receive result"""
        # Upload data
        upload_time, upload_energy = self.calculate_transmission(
            task.input_size, self.cloud_bandwidth, 
            self.cloud_latency_base, self.cloud_efficiency
        )
        
        # Download result
        download_time, download_energy = self.calculate_transmission(
            task.output_size, self.cloud_bandwidth,
            self.cloud_latency_base, self.cloud_efficiency
        )
        
        total_time = upload_time + download_time
        total_energy = upload_energy + download_energy
        
        self.logger.debug(f"Cloud transmission - Time: {total_time:.4f}s, "
                         f"Energy: {total_energy:.4f}J")
        
        return total_time, total_energy, self.cloud_latency_base
    
    def get_statistics(self) -> Dict:
        """Get network statistics"""
        return {
            'total_data_transmitted_mb': self.total_data_transmitted,
            'total_transmission_time': self.total_transmission_time,
            'total_transmission_energy': self.total_transmission_energy
        }

# ============================================================================
# MAIN SIMULATOR
# ============================================================================

class YAFSSimulator:
    """Main simulator class coordinating all components"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize nodes
        self.device = Device()
        self.edge = EdgeNode()
        self.cloud = CloudNode()
        self.network = NetworkSimulator()
        
        # Simulation state
        self.current_time = 0.0
        self.task_counter = 0
        
        # Results tracking
        self.task_results: List[TaskResult] = []
        self.execution_log = []
        
        self.logger.info("YAFS Simulator initialized")
    
    def create_task(self, input_size: float, output_size: float,
                   cpu_cycles: float, deadline: float, 
                   priority: str = "medium") -> TaskDefinition:
        """Create a new task"""
        self.task_counter += 1
        task = TaskDefinition(
            task_id=self.task_counter,
            input_size=input_size,
            output_size=output_size,
            cpu_cycles=cpu_cycles,
            deadline=deadline,
            priority=priority,
            arrival_time=self.current_time
        )
        self.logger.info(f"Created task {task.task_id}: {cpu_cycles}M cycles, "
                        f"deadline {deadline}s")
        return task
    
    def run_locally(self, task: TaskDefinition) -> TaskResult:
        """Execute task locally on device"""
        self.logger.info(f"Task {task.task_id} → Device (local execution)")
        
        # Execute on device
        result = self.device.execute_task(task, self.current_time)
        
        # Update simulation time
        self.current_time = result.end_time
        
        # Log result
        self.task_results.append(result)
        self._log_execution(task, result)
        
        return result
    
    def offload_to_edge(self, task: TaskDefinition) -> TaskResult:
        """Offload task to edge node"""
        self.logger.info(f"Task {task.task_id} → Edge (offloading)")
        
        start_time = self.current_time
        
        # Network transmission
        net_time, net_energy, prop_delay = self.network.transmit_to_edge(task)
        
        # Device idle energy during transmission and computation
        self.device.consume_battery(net_energy)
        
        # Execute on edge
        edge_result = self.edge.execute_task(task, self.current_time)
        
        # Device idle during edge computation
        idle_time = edge_result.computation_time + prop_delay
        idle_energy = self.device.power_idle * idle_time
        self.device.consume_battery(idle_energy)
        
        # Total metrics
        total_latency = net_time + edge_result.computation_time
        total_energy = net_energy + idle_energy
        
        # Update device statistics
        self.device.total_energy += total_energy
        
        # Create result
        result = TaskResult(
            task_id=task.task_id,
            location=NodeType.EDGE,
            status=TaskStatus.COMPLETED,
            energy_consumed=total_energy,
            latency=total_latency,
            start_time=start_time,
            end_time=start_time + total_latency,
            deadline_met=total_latency <= task.deadline,
            network_delay=net_time,
            computation_time=edge_result.computation_time
        )
        
        # Update simulation time
        self.current_time = result.end_time
        
        # Log result
        self.task_results.append(result)
        self._log_execution(task, result)
        
        return result
    
    def offload_to_cloud(self, task: TaskDefinition) -> TaskResult:
        """Offload task to cloud node"""
        self.logger.info(f"Task {task.task_id} → Cloud (offloading)")
        
        start_time = self.current_time
        
        # Network transmission
        net_time, net_energy, prop_delay = self.network.transmit_to_cloud(task)
        
        # Device idle energy during transmission
        self.device.consume_battery(net_energy)
        
        # Execute on cloud
        cloud_result = self.cloud.execute_task(task, self.current_time)
        
        # Device idle during cloud computation
        idle_time = cloud_result.computation_time + prop_delay + self.cloud.queue_delay
        idle_energy = self.device.power_idle * idle_time
        self.device.consume_battery(idle_energy)
        
        # Total metrics
        total_latency = net_time + cloud_result.computation_time + self.cloud.queue_delay
        total_energy = net_energy + idle_energy
        
        # Update device statistics
        self.device.total_energy += total_energy
        
        # Create result
        result = TaskResult(
            task_id=task.task_id,
            location=NodeType.CLOUD,
            status=TaskStatus.COMPLETED,
            energy_consumed=total_energy,
            latency=total_latency,
            start_time=start_time,
            end_time=start_time + total_latency,
            deadline_met=total_latency <= task.deadline,
            network_delay=net_time,
            computation_time=cloud_result.computation_time
        )
        
        # Update simulation time
        self.current_time = result.end_time
        
        # Log result
        self.task_results.append(result)
        self._log_execution(task, result)
        
        return result
    
    def _log_execution(self, task: TaskDefinition, result: TaskResult):
        """Log task execution details"""
        log_entry = {
            'time': self.current_time,
            'task': task.to_dict(),
            'result': result.to_dict(),
            'device_battery': self.device.battery_level,
            'device_load': self.device.current_load,
            'edge_load': self.edge.current_load,
            'cloud_load': self.cloud.current_load
        }
        self.execution_log.append(log_entry)
        
        self.logger.info(
            f"Task {task.task_id} completed: "
            f"Location={result.location.value}, "
            f"Energy={result.energy_consumed:.4f}J, "
            f"Latency={result.latency:.4f}s, "
            f"Deadline={'MET' if result.deadline_met else 'MISSED'}, "
            f"Battery={self.device.battery_level:.2f}%"
        )
    
    def get_system_state(self) -> Dict:
        """Get current system state (for APEATO algorithm)"""
        return {
            'battery': self.device.battery_level,
            'cpu_load': self.device.current_load,
            'bandwidth': self.network.edge_bandwidth * 1e6,  # Convert to bps
            'time_of_day': int((self.current_time / 3600) % 24),  # Hour of day
            'device_load': self.device.current_load,
            'edge_load': self.edge.current_load,
            'cloud_load': self.cloud.current_load,
            'network_reliability_edge': 0.95,
            'network_reliability_cloud': 0.90
        }
    
    def get_statistics(self) -> Dict:
        """Get comprehensive simulation statistics"""
        total_tasks = len(self.task_results)
        
        if total_tasks == 0:
            return {
                'total_tasks': 0,
                'message': 'No tasks executed yet'
            }
        
        # Task distribution
        device_tasks = sum(1 for r in self.task_results if r.location == NodeType.DEVICE)
        edge_tasks = sum(1 for r in self.task_results if r.location == NodeType.EDGE)
        cloud_tasks = sum(1 for r in self.task_results if r.location == NodeType.CLOUD)
        
        # Energy statistics
        total_energy = sum(r.energy_consumed for r in self.task_results)
        avg_energy = total_energy / total_tasks
        
        # Latency statistics
        total_latency = sum(r.latency for r in self.task_results)
        avg_latency = total_latency / total_tasks
        
        # Deadline statistics
        deadlines_met = sum(1 for r in self.task_results if r.deadline_met)
        deadline_success_rate = (deadlines_met / total_tasks) * 100
        
        stats = {
            'simulation_time': self.current_time,
            'total_tasks': total_tasks,
            'task_distribution': {
                'device': device_tasks,
                'edge': edge_tasks,
                'cloud': cloud_tasks
            },
            'energy': {
                'total_joules': total_energy,
                'average_per_task': avg_energy,
                'battery_consumed_percent': self.device.initial_battery - self.device.battery_level
            },
            'latency': {
                'total_seconds': total_latency,
                'average_per_task': avg_latency
            },
            'deadline_performance': {
                'total_met': deadlines_met,
                'total_missed': total_tasks - deadlines_met,
                'success_rate_percent': deadline_success_rate
            },
            'device_status': self.device.get_status(),
            'edge_status': self.edge.get_status(),
            'cloud_status': self.cloud.get_status(),
            'network_statistics': self.network.get_statistics()
        }
        
        return stats
    
    def print_statistics(self):
        """Print formatted statistics"""
        stats = self.get_statistics()
        
        print("\n" + "="*70)
        print("YAFS SIMULATOR - EXECUTION STATISTICS")
        print("="*70)
        
        print(f"\n📊 GENERAL")
        print(f"  Simulation Time: {stats['simulation_time']:.4f}s")
        print(f"  Total Tasks: {stats['total_tasks']}")
        
        print(f"\n📍 TASK DISTRIBUTION")
        dist = stats['task_distribution']
        print(f"  Device (Local): {dist['device']} tasks")
        print(f"  Edge: {dist['edge']} tasks")
        print(f"  Cloud: {dist['cloud']} tasks")
        
        print(f"\n⚡ ENERGY CONSUMPTION")
        energy = stats['energy']
        print(f"  Total Energy: {energy['total_joules']:.4f} J")
        print(f"  Average per Task: {energy['average_per_task']:.4f} J")
        print(f"  Battery Consumed: {energy['battery_consumed_percent']:.2f}%")
        print(f"  Battery Remaining: {stats['device_status']['battery_level']:.2f}%")
        
        print(f"\n⏱️  LATENCY")
        latency = stats['latency']
        print(f"  Total Latency: {latency['total_seconds']:.4f}s")
        print(f"  Average per Task: {latency['average_per_task']:.4f}s")
        
        print(f"\n🎯 DEADLINE PERFORMANCE")
        deadline = stats['deadline_performance']
        print(f"  Met: {deadline['total_met']}")
        print(f"  Missed: {deadline['total_missed']}")
        print(f"  Success Rate: {deadline['success_rate_percent']:.2f}%")
        
        print(f"\n🖥️  NODE STATUS")
        print(f"  Device Load: {stats['device_status']['load']:.2f}%")
        print(f"  Edge Load: {stats['edge_status']['load']:.2f}%")
        print(f"  Cloud Load: {stats['cloud_status']['load']:.2f}%")
        
        print("\n" + "="*70 + "\n")
    
    def export_results(self, filename: str = "simulation_results.json"):
        """Export results to JSON file"""
        data = {
            'statistics': self.get_statistics(),
            'execution_log': self.execution_log
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Results exported to {filename}")
    
    def reset(self):
        """Reset simulator to initial state"""
        self.device = Device()
        self.edge = EdgeNode()
        self.cloud = CloudNode()
        self.network = NetworkSimulator()
        self.current_time = 0.0
        self.task_counter = 0
        self.task_results = []
        self.execution_log = []
        self.logger.info("Simulator reset")


# ============================================================================
# EXAMPLE: BASIC USAGE WITHOUT APEATO
# ============================================================================

def example_basic_usage():
    """Example of basic simulator usage"""
    print("\n" + "="*70)
    print("EXAMPLE 1: BASIC SIMULATOR USAGE")
    print("="*70 + "\n")
    
    # Create simulator
    sim = YAFSSimulator()
    
    # Create tasks
    task1 = sim.create_task(
        input_size=10,  # 10 MB
        output_size=1,  # 1 MB
        cpu_cycles=5000,  # 5000M cycles
        deadline=2.0,  # 2 seconds
        priority="high"
    )
    
    task2 = sim.create_task(
        input_size=5,
        output_size=0.5,
        cpu_cycles=2000,
        deadline=1.0,
        priority="medium"
    )
    
    task3 = sim.create_task(
        input_size=20,
        output_size=2,
        cpu_cycles=10000,
        deadline=5.0,
        priority="low"
    )
    
    # Execute tasks on different nodes
    result1 = sim.run_locally(task1)
    result2 = sim.offload_to_edge(task2)
    result3 = sim.offload_to_cloud(task3)
    
    # Print statistics
    sim.print_statistics()


# ============================================================================
# EXAMPLE: INTEGRATION WITH APEATO ALGORITHM
# ============================================================================

from APEATO import APEATOAlgorithm, Task as APEATOTask, SystemState, Activity, Priority as APEATOPriority, Location

def example_with_apeato():
    """Example showing APEATO algorithm integration"""
    print("\n" + "="*70)
    print("EXAMPLE 2: APEATO ALGORITHM INTEGRATION")
    print("="*70 + "\n")
    
    # Create simulator and APEATO algorithm
    sim = YAFSSimulator()
    apeato = APEATOAlgorithm()
    
    print("🤖 APEATO Algorithm initialized\n")
    
    # Create multiple tasks
    tasks = []
    for i in range(10):
        task = sim.create_task(
            input_size=random.uniform(5, 15),
            output_size=random.uniform(0.5, 2),
            cpu_cycles=random.uniform(1000, 8000),
            deadline=random.uniform(1.0, 5.0),
            priority=random.choice(["low", "medium", "high"])
        )
        tasks.append(task)
    
    print(f"📋 Created {len(tasks)} tasks\n")
    
    # Process each task using APEATO
    for task in tasks:
        # Get current system state
        sys_state = sim.get_system_state()
        
        # Convert priority
        priority_map = {
            "low": APEATOPriority.LOW,
            "medium": APEATOPriority.MEDIUM,
            "high": APEATOPriority.HIGH
        }
        
        # Convert activity based on device load
        if sys_state['device_load'] < 30:
            activity = Activity.IDLE
        elif sys_state['device_load'] < 60:
            activity = Activity.NORMAL
        elif sys_state['device_load'] < 80:
            activity = Activity.ACTIVE
        else:
            activity = Activity.GAMING
        
        # Create APEATO task
        apeato_task = APEATOTask(
            task_id=task.task_id,
            workload=task.cpu_cycles * 1e6,  # Convert to cycles
            data_size=task.input_size * 8e6,  # Convert MB to bits
            result_size=task.output_size * 8e6,
            priority=priority_map[task.priority],
            deadline=task.deadline
        )
        
        # Create APEATO system state
        apeato_state = SystemState(
            battery=sys_state['battery'],
            cpu_load=sys_state['cpu_load'],
            bandwidth=sys_state['bandwidth'],
            time_of_day=sys_state['time_of_day'],
            activity=activity
        )
        
        # APEATO makes decision
        decision = apeato.make_decision(apeato_task, apeato_state)
        
        # Execute based on APEATO decision
        if decision == Location.DEVICE:
            result = sim.run_locally(task)
        elif decision == Location.EDGE:
            result = sim.offload_to_edge(task)
        else:  # CLOUD
            result = sim.offload_to_cloud(task)
        
        # Provide feedback to APEATO for learning
        apeato.execute_and_learn(apeato_task, apeato_state, decision)
    
    # Print final statistics
    sim.print_statistics()
    
    # Export results
    sim.export_results("apeato_simulation_results.json")
    print("✅ Results exported to 'apeato_simulation_results.json'\n")


# ============================================================================
# EXAMPLE: COMPARISON - WITH AND WITHOUT APEATO
# ============================================================================

def example_comparison():
    """Compare APEATO vs baseline strategies"""
    print("\n" + "="*70)
    print("EXAMPLE 3: PERFORMANCE COMPARISON")
    print("="*70 + "\n")
    
    try:
        from APEATO import APEATOAlgorithm, Task as APEATOTask, SystemState, Activity, Priority as APEATOPriority, Location
    except ImportError:
        print("⚠️  APEATO algorithm module not found.")
        return
    
    # Test scenarios
    test_tasks = []
    for i in range(20):
        test_tasks.append({
            'input_size': random.uniform(5, 20),
            'output_size': random.uniform(0.5, 3),
            'cpu_cycles': random.uniform(1000, 10000),
            'deadline': random.uniform(1.0, 6.0),
            'priority': random.choice(["low", "medium", "high"])
        })
    
    results_comparison = {}
    
    # Strategy 1: All local
    print("🔄 Testing Strategy 1: All Local Execution\n")
    sim1 = YAFSSimulator()
    for task_def in test_tasks:
        task = sim1.create_task(**task_def)
        sim1.run_locally(task)
    stats1 = sim1.get_statistics()
    results_comparison['all_local'] = stats1
    
    # Strategy 2: All edge
    print("🔄 Testing Strategy 2: All Edge Offloading\n")
    sim2 = YAFSSimulator()
    for task_def in test_tasks:
        task = sim2.create_task(**task_def)
        sim2.offload_to_edge(task)
    stats2 = sim2.get_statistics()
    results_comparison['all_edge'] = stats2
    
    # Strategy 3: All cloud
    print("🔄 Testing Strategy 3: All Cloud Offloading\n")
    sim3 = YAFSSimulator()
    for task_def in test_tasks:
        task = sim3.create_task(**task_def)
        sim3.offload_to_cloud(task)
    stats3 = sim3.get_statistics()
    results_comparison['all_cloud'] = stats3
    
    # Strategy 4: APEATO
    print("🔄 Testing Strategy 4: APEATO Algorithm\n")
    sim4 = YAFSSimulator()
    apeato = APEATOAlgorithm()
    
    for task_def in test_tasks:
        task = sim4.create_task(**task_def)
        sys_state = sim4.get_system_state()
        
        priority_map = {"low": APEATOPriority.LOW, "medium": APEATOPriority.MEDIUM, "high": APEATOPriority.HIGH}
        activity = Activity.NORMAL if sys_state['device_load'] < 60 else Activity.ACTIVE
        
        apeato_task = APEATOTask(
            task_id=task.task_id,
            workload=task.cpu_cycles * 1e6,
            data_size=task.input_size * 8e6,
            result_size=task.output_size * 8e6,
            priority=priority_map[task.priority],
            deadline=task.deadline
        )
        
        apeato_state = SystemState(
            battery=sys_state['battery'],
            cpu_load=sys_state['cpu_load'],
            bandwidth=sys_state['bandwidth'],
            time_of_day=sys_state['time_of_day'],
            activity=activity
        )
        
        decision = apeato.make_decision(apeato_task, apeato_state)
        
        if decision == Location.DEVICE:
            sim4.run_locally(task)
        elif decision == Location.EDGE:
            sim4.offload_to_edge(task)
        else:
            sim4.offload_to_cloud(task)
        
        apeato.execute_and_learn(apeato_task, apeato_state, decision)
    
    stats4 = sim4.get_statistics()
    results_comparison['apeato'] = stats4
    
    # Print comparison
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON RESULTS")
    print("="*70 + "\n")
    
    strategies = [
        ('All Local', stats1),
        ('All Edge', stats2),
        ('All Cloud', stats3),
        ('APEATO', stats4)
    ]
    
    print(f"{'Strategy':<15} {'Avg Energy (J)':<15} {'Avg Latency (s)':<17} {'Battery Left (%)':<17} {'Deadlines Met (%)':<20}")
    print("-" * 84)
    
    for name, stats in strategies:
        avg_energy = stats['energy']['average_per_task']
        avg_latency = stats['latency']['average_per_task']
        battery_left = stats['device_status']['battery_level']
        deadline_rate = stats['deadline_performance']['success_rate_percent']
        
        print(f"{name:<15} {avg_energy:<15.4f} {avg_latency:<17.4f} {battery_left:<17.2f} {deadline_rate:<20.2f}")
    
    print("\n" + "="*70)
    
    # Calculate improvements
    print("\n📈 APEATO IMPROVEMENTS OVER BASELINES:\n")
    
    baseline_energy = stats1['energy']['average_per_task']
    apeato_energy = stats4['energy']['average_per_task']
    energy_improvement = ((baseline_energy - apeato_energy) / baseline_energy) * 100
    
    baseline_battery = 100 - stats1['energy']['battery_consumed_percent']
    apeato_battery = stats4['device_status']['battery_level']
    battery_improvement = ((apeato_battery - baseline_battery) / baseline_battery) * 100
    
    print(f"  Energy Efficiency: {energy_improvement:+.2f}%")
    print(f"  Battery Life Extension: {battery_improvement:+.2f}%")
    print(f"  Deadline Success Rate: {stats4['deadline_performance']['success_rate_percent']:.2f}%")
    
    print("\n")


# ============================================================================
# EXAMPLE: CUSTOM OFFLOADING ALGORITHM TEMPLATE
# ============================================================================

class CustomOffloadingAlgorithm:
    """Template for creating custom offloading algorithms"""
    
    def __init__(self, simulator: YAFSSimulator):
        self.sim = simulator
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def decide(self, task: TaskDefinition) -> str:
        """
        Make offloading decision based on custom logic
        
        Returns: 'local', 'edge', or 'cloud'
        """
        # Get current system state
        state = self.sim.get_system_state()
        
        # Example simple logic: battery-aware decisions
        if state['battery'] < 20:
            # Low battery: prefer offloading
            if task.cpu_cycles > 5000:
                return 'cloud'
            else:
                return 'edge'
        elif state['battery'] > 70:
            # High battery: can process locally
            if task.cpu_cycles < 3000:
                return 'local'
            else:
                return 'edge'
        else:
            # Medium battery: balanced approach
            if task.deadline < 1.0:
                return 'cloud'  # Fast execution needed
            elif task.cpu_cycles < 4000:
                return 'local'
            else:
                return 'edge'
    
    def execute(self, task: TaskDefinition) -> TaskResult:
        """Execute task based on decision"""
        decision = self.decide(task)
        
        if decision == 'local':
            return self.sim.run_locally(task)
        elif decision == 'edge':
            return self.sim.offload_to_edge(task)
        else:
            return self.sim.offload_to_cloud(task)


def example_custom_algorithm():
    """Example using custom offloading algorithm"""
    print("\n" + "="*70)
    print("EXAMPLE 4: CUSTOM OFFLOADING ALGORITHM")
    print("="*70 + "\n")
    
    sim = YAFSSimulator()
    algorithm = CustomOffloadingAlgorithm(sim)
    
    # Create and process tasks
    for i in range(15):
        task = sim.create_task(
            input_size=random.uniform(5, 15),
            output_size=random.uniform(0.5, 2),
            cpu_cycles=random.uniform(1000, 8000),
            deadline=random.uniform(0.5, 4.0),
            priority=random.choice(["low", "medium", "high"])
        )
        
        # Use custom algorithm to decide and execute
        result = algorithm.execute(task)
    
    sim.print_statistics()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("YAFS (Yet Another Fog Simulator)")
    print("Integrated with APEATO Task Offloading Algorithm")
    print("="*70)
    
    # Run examples
    example_basic_usage()
    
    print("\n" + "="*70)
    print("Press Enter to continue with APEATO integration example...")
    print("="*70)
    input()
    
    example_with_apeato()
    
    print("\n" + "="*70)
    print("Press Enter to continue with performance comparison...")
    print("="*70)
    input()
    
    example_comparison()
    
    print("\n" + "="*70)
    print("Press Enter to continue with custom algorithm example...")
    print("="*70)
    input()
    
    example_custom_algorithm()
    
    print("\n✅ All examples completed successfully!\n")