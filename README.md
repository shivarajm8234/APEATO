# APEATO: Intelligent Task Offloading in Fog–Edge–Cloud Environments

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🚀 Overview

**APEATO (Adaptive Priority and Energy-Aware Task Offloading)** is an intelligent task orchestration algorithm designed for heterogeneous Fog-Edge-Cloud computing environments. The system dynamically decides where to execute IoT tasks (Local Device, Edge Node, or Cloud Server) to optimize both Quality of Service (QoS) and energy efficiency.

### Key Features

- ✅ **Multi-Objective Optimization**: Balances energy consumption and latency constraints
- ✅ **Adaptive Decision Making**: Dynamically adjusts priorities based on battery level, network conditions, and task deadlines
- ✅ **Real-Time Context Awareness**: Monitors network bandwidth, device state, and node availability
- ✅ **Priority-Based Classification**: Categorizes tasks into Heavy/Critical, Medium, and Light tiers
- ✅ **Comprehensive Simulation**: Includes 10,000+ synthetic task instances for evaluation

---

## 📋 Table of Contents

- [Problem Statement](#problem-statement)
- [Why Not Classical CPU Scheduling?](#why-not-classical-cpu-scheduling)
- [APEATO Algorithm](#apeato-algorithm)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Results](#results)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [References](#references)

---

## 🎯 Problem Statement

Modern IoT applications face critical challenges in distributed computing environments:

1. **QoS Violations**: Strict latency deadlines for real-time applications (autonomous driving, smart healthcare)
2. **Battery Depletion**: Computationally intensive tasks drain mobile device batteries rapidly
3. **Stochastic Workloads**: Unpredictable task arrival rates and fluctuating network conditions
4. **Resource Heterogeneity**: Devices with vastly different computational capabilities

---

## ❌ Why Not Classical CPU Scheduling?

Traditional OS scheduling algorithms (FCFS, SJF, Round Robin, Priority Scheduling) are fundamentally inadequate for fog computing:

| Classical Scheduling | Fog Offloading Requirements |
|---------------------|----------------------------|
| ❌ No network awareness | ✅ Data transmission time is critical |
| ❌ No energy metrics | ✅ Battery life is paramount |
| ❌ Centralized single-machine queue | ✅ Distributed multi-node decision making |
| ❌ Optimizes CPU time only | ✅ Requires holistic cost function |

---

## 🧠 APEATO Algorithm

The algorithm operates in **four phases**:

### Phase 1: Task Profiling & Priority Classification
Tasks are classified based on workload, data size, and deadline:
- **Heavy/Critical**: High computational demand, strict deadline
- **Medium**: Moderate demand, flexible deadline
- **Light**: Low demand, delay-tolerant

### Phase 2: Real-Time Context Monitoring
Continuously monitors:
- Network bandwidth and RTT
- Device battery level and CPU load
- Edge node and cloud server availability

### Phase 3: Multi-Objective Cost Function
Calculates weighted cost for each execution venue:

```
Cost_V = α · E_V + (1 - α) · T_V + P_V
```

Where:
- `E_V`: Estimated energy consumption
- `T_V`: Estimated total time (transmission + processing)
- `α`: Adaptive weighting factor (0 ≤ α ≤ 1)
- `P_V`: Penalty for congestion or deadline violations

### Phase 4: Decision & Offloading
Selects the venue with **minimum cost**:
- **Local**: High transmission cost or light tasks
- **Edge**: Latency-sensitive tasks with sufficient edge resources
- **Cloud**: Extremely heavy tasks requiring massive processing power

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     IoT/Device Layer                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Task Generator│  │ APEATO Client│  │Local Executor│     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                     Fog/Edge Layer                          │
│  ┌──────────────┐  ┌──────────────┐                        │
│  │Resource Monitor│ │  Edge Server │                        │
│  └──────────────┘  └──────────────┘                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                      Cloud Layer                            │
│            ┌──────────────────────┐                         │
│            │ Cloud Data Center    │                         │
│            └──────────────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

**Decision Engine Components:**
- **Profiler**: Extracts task attributes
- **State Manager**: Aggregates network and device metrics
- **Optimizer**: Runs APEATO cost function
- **Dispatcher**: Routes task data to selected venue

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd os
```

2. **Create virtual environment** (recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

If `requirements.txt` doesn't exist, install core dependencies:
```bash
pip install numpy pandas matplotlib seaborn networkx
```

---

## 🎮 Usage

### 1. Run APEATO Simulation

```bash
python APEATO.py
```

### 2. Run GUI Simulator

```bash
python fog_simulator_gui.py
```

### 3. Test APEATO Algorithm

```bash
python test_apeato.py
```

### 4. Generate Comparison Results

```bash
python comparision_file.py
```

### 5. Run Sensitivity Analysis

```bash
python sens.py
```

### 6. Generate Synthetic Dataset

```bash
python syntethic.py
```

---

## 📊 Dataset

The project uses `diverse_tasks.csv` containing **10,000+ task instances** with the following attributes:

| Attribute | Description | Unit |
|-----------|-------------|------|
| `task_id` | Unique identifier | - |
| `workload` | CPU cycles required | CPU Cycles |
| `data_size` | Data volume to transmit | MB |
| `result_size` | Result data volume | MB |
| `priority` | Task importance (0-2) | 0: Low, 1: Medium, 2: High |
| `deadline` | Time constraint | Seconds |
| `type` | Task classification | Light/Medium/Heavy |

> **Note**: The dataset explicitly excludes classical OS metrics (waiting time, turnaround time) as they are irrelevant to distributed offloading decisions.

---

## 📈 Results

Expected performance improvements over baseline methods (Random, Greedy, Static Offloading):

- **Energy Efficiency**: 20-30% reduction in average energy consumption
- **Deadline Hit Rate**: Significant increase in tasks meeting deadlines
- **Load Balancing**: Improved resource utilization across edge nodes
- **Adaptability**: Robust performance under network degradation

Results are stored in the `results/` directory with visualizations and performance metrics.

---

## 📁 Project Structure

```
os/
├── APEATO.py                 # Main APEATO algorithm implementation
├── YAFS.py                   # YAFS framework integration
├── fog_simulator_gui.py      # GUI-based simulator
├── test_apeato.py           # Unit tests for APEATO
├── comparision_file.py      # Baseline comparison scripts
├── sens.py                  # Sensitivity analysis
├── syntethic.py             # Synthetic dataset generator
├── create_cpu_task.py       # CPU task creation utilities
├── diverse_tasks.csv        # Main dataset (10,000+ tasks)
├── tasks.csv                # Additional task dataset
├── project_synopsis.md      # Detailed project documentation
├── results/                 # Simulation results and graphs
├── results_sensitivity/     # Sensitivity analysis results
├── images/                  # Visualization assets
├── YAFS/                    # YAFS framework files
└── venv/                    # Python virtual environment
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📚 References

1. **Bonomi, F., Milito, R., Zhu, J., & Addepalli, S. (2012).** "Fog Computing and Its Role in the Internet of Things." *Proceedings of the First Edition of the MCC Workshop on Mobile Cloud Computing*, ACM, pp. 13-16.

2. **Mao, Y., You, C., Zhang, J., Huang, K., & Letaief, K. B. (2017).** "A Survey on Mobile Edge Computing: The Communication Perspective." *IEEE Communications Surveys & Tutorials*, 19(4), 2322-2358.

3. **Silberschatz, A., Galvin, P. B., & Gagne, G. (2018).** *Operating System Concepts*, 10th Edition. Wiley.

4. **Mahmud, R., Kotagiri, R., & Buyya, R. (2018).** "Fog Computing: A Taxonomy, Survey and Future Directions." *Internet of Things*, Elsevier, pp. 103-130.

5. **Zhang, Y., & Ansari, N. (2020).** "Offloading in Mobile Edge Computing with Energy-Delay Trade-off." *IEEE Internet of Things Journal*, 7(4), 3220-3231.

6. **Kumar, K., & Lu, Y. H. (2010).** "Cloud Computing for Mobile Users: Can Offloading Computation Save Energy?" *Computer*, 43(4), 51-56.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

Final Year Engineering Project  
**Topic**: Intelligent Task Offloading in Fog–Edge–Cloud Environments using the APEATO Algorithm

---

## 🙏 Acknowledgments

- YAFS (Yet Another Fog Simulator) framework
- Academic references and research papers cited above
- Open-source Python community

---

## 📧 Contact

For questions or collaboration opportunities, please open an issue in the repository.

---

**⭐ If you find this project useful, please consider giving it a star!**
