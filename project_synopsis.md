# Project Synopsis: Intelligent Task Offloading in Fog–Edge–Cloud Environments using the APEATO Algorithm

## 1. Introduction
The advent of the Internet of Things (IoT) has catalyzed a paradigm shift in computing, generating unprecedented volumes of data at the network edge. While Cloud Computing provides massive processing power, its centralized nature introduces significant latency and bandwidth bottlenecks, rendering it unsuitable for real-time, delay-sensitive applications such as autonomous driving, smart healthcare, and industrial automation. To mitigate these challenges, **Fog and Edge Computing** have emerged as complementary architectures that extend cloud capabilities closer to the data source.

However, the Fog-Edge-Cloud continuum is characterized by extreme heterogeneity, resource constraints, and dynamic network conditions. IoT devices have limited battery life and processing power, while fog nodes exhibit varying computational capacities. Consequently, the naive execution of tasks—either purely locally or purely on the cloud—leads to suboptimal performance. There is a critical need for **intelligent task offloading orchestration** that can dynamically decide *where* to execute a task (Local Device, Edge Node, or Cloud Server) to optimize Quality of Service (QoS) and energy efficiency. This project proposes the **APEATO (Adaptive Priority and Energy-Aware Task Offloading)** algorithm to address these challenges through a multi-objective optimization approach.

## 2. Problem Statement
Despite the potential of fog computing, efficient task offloading remains a formidable challenge due to the stochastic nature of IoT environments. The core problems addressed by this project include:

*   **QoS Violations in Latency-Critical Applications:** Mobile applications often have strict deadlines. High network latency to the cloud or queuing delays at overloaded edge nodes can cause deadline misses, leading to service failure.
*   **Battery Depletion in Mobile Devices:** Computationally intensive tasks (e.g., Deep Learning inference) drain device batteries rapidly. However, offloading also consumes transmission energy. A poor trade-off decision can exacerbate energy consumption.
*   **Stochastic Workload & Network Instability:** Task arrival rates are unpredictable, and wireless channel conditions (bandwidth, interference) fluctuate dynamically. Static offloading policies fail to adapt to these variations, leading to load imbalances and system instability.
*   **Resource Heterogeneity:** The computing environment consists of devices with vastly different capabilities. A "one-size-fits-all" scheduling approach is ineffective in such a diverse landscape.

## 3. Why Classical CPU Scheduling Cannot Solve This Problem
Classical Operating System (OS) CPU scheduling algorithms—such as **First-Come-First-Serve (FCFS)**, **Shortest Job First (SJF)**, **Round Robin (RR)**, and **Priority Scheduling**—are fundamentally inadequate for the distributed fog computing paradigm. Their limitations are threefold:

1.  **Lack of Network Awareness:** Classical schedulers optimize for *local* CPU time (metrics: arrival time, burst time, waiting time, turnaround time). They assume zero communication cost between processes. In fog offloading, **data transmission time** is often the dominant factor, which classical algorithms completely ignore.
2.  **Absence of Energy Metrics:** Standard CPU schedulers aim to maximize throughput or minimize response time, with no consideration for **energy consumption**. In IoT scenarios, extending battery life is often more critical than raw speed, a dimension absent in classical scheduling logic.
3.  **Distributed vs. Centralized Scope:** Classical algorithms assume a centralized queue on a single machine. Fog offloading involves a **distributed decision-making process** across multiple distinct physical nodes (Device, Edge, Cloud) with varying architectures. Metrics like "waiting time in the ready queue" do not translate to "network propagation delay" or "remote processing cost."

Therefore, a specialized offloading algorithm like APEATO is required to handle the specific constraints of data size, transmission power, and remote node availability.

## 4. Existing Techniques & Literature Survey
Current research in fog task offloading can be broadly categorized, each with specific limitations:

*   **Static & Heuristic Approaches:**
    *   *Technique:* Fixed threshold-based rules (e.g., "Offload if task size > 5MB").
    *   *Limitation:* Rigid and inflexible. They fail to adapt to real-time changes in network bandwidth or node congestion, leading to poor performance in dynamic scenarios.
*   **Optimization-Based Methods (ILP/Game Theory):**
    *   *Technique:* Formulating the problem as an Integer Linear Programming (ILP) or Nash Equilibrium problem.
    *   *Limitation:* These methods often seek a global optimum, which is **NP-hard**. The high computational complexity makes them unsuitable for real-time, sub-second decision-making on resource-constrained devices.
*   **Deep Reinforcement Learning (DRL) Schedulers:**
    *   *Technique:* Using Neural Networks to learn offloading policies (e.g., DQN, DDPG).
    *   *Limitation:* While powerful, DRL models suffer from **slow convergence** (long training times), high inference overhead, and the "cold start" problem, where performance is poor until sufficient data is collected.
*   **Energy-Only Minimization:**
    *   *Technique:* Algorithms focused solely on saving battery.
    *   *Limitation:* Often ignore strict latency deadlines, rendering them unusable for time-critical applications like AR/VR or safety systems.

## 5. Proposed Solution – APEATO Algorithm
The **APEATO (Adaptive Priority and Energy-Aware Task Offloading)** algorithm is designed to overcome the shortcomings of existing methods by balancing conflicting objectives (Energy vs. Latency) in real-time. The algorithm operates in four distinct phases:

### Phase 1: Task Profiling & Priority Classification
Incoming tasks are analyzed based on their metadata (Workload, Data Size, Deadline). They are classified into three priority tiers:
*   **Heavy/Critical:** High computational demand, strict deadline (e.g., real-time video processing).
*   **Medium:** Moderate demand, flexible deadline.
*   **Light:** Low demand, delay-tolerant (e.g., background data logging).

### Phase 2: Real-Time Context Monitoring
The system continuously monitors the dynamic state of the environment:
*   **Network Status:** Available uplink/downlink bandwidth and round-trip time (RTT).
*   **Device State:** Current battery level and local CPU load.
*   **Node Availability:** Resource utilization of nearby Edge nodes and Cloud server status.

### Phase 3: Multi-Objective Cost Function Optimization
APEATO calculates a **Weighted Cost** for executing the task on each potential venue ($V \in \{Local, Edge, Cloud\}$). The cost function is defined as:
$$ Cost_V = \alpha \cdot E_V + (1 - \alpha) \cdot T_V + P_V $$
Where:
*   $E_V$: Estimated Energy Consumption on venue $V$.
*   $T_V$: Estimated Total Time (Transmission + Processing) on venue $V$.
*   $\alpha$: **Adaptive Weighting Factor** ($0 \le \alpha \le 1$). This factor dynamically shifts based on context. If battery is low, $\alpha$ increases to prioritize Energy. If deadline is tight, $\alpha$ decreases to prioritize Time.
*   $P_V$: **Penalty Function**. Adds a high cost to venues that are congested or likely to miss the deadline.

### Phase 4: Decision & Offloading
The algorithm selects the venue with the **minimum Cost**.
*   *Local Execution:* Selected if transmission cost is high or task is light.
*   *Edge Offloading:* Selected for latency-sensitive tasks where Edge resources are sufficient.
*   *Cloud Offloading:* Selected for extremely heavy tasks where local/edge processing would exceed deadlines despite the network delay.

## 6. Dataset Description
The project utilizes a synthetic but realistic dataset (`diverse_tasks.csv`) designed to model a heterogeneous fog environment. It contains **10,000+ task instances** with the following attributes:
*   **`task_id`**: Unique identifier.
*   **`workload` (CPU Cycles):** The number of CPU instructions required to complete the task.
*   **`data_size` (MB):** The volume of data to be transmitted to the offloading node.
*   **`result_size` (MB):** The volume of result data to be downloaded.
*   **`priority` (0-2):** Task importance level (0: Low, 1: Medium, 2: High).
*   **`deadline` (Seconds):** The strict time constraint for task completion.
*   **`type` (Categorical):** 'Light', 'Medium', 'Heavy' classification.

*Note: The dataset explicitly excludes classical OS metrics (waiting time, etc.) as they are irrelevant to the distributed offloading decision matrix.*

## 7. System Architecture
The system is modeled as a three-tier architecture:

1.  **IoT/Device Layer:**
    *   **Task Generator:** Simulates application requests.
    *   **APEATO Client:** Runs the lightweight decision engine.
    *   **Local Executor:** Processes tasks locally if selected.
2.  **Fog/Edge Layer:**
    *   **Resource Monitor:** Broadcasts current load and bandwidth availability.
    *   **Edge Server:** Executes offloaded tasks with moderate latency.
3.  **Cloud Layer:**
    *   **Cloud Data Center:** Handles massive workloads that exceed Edge capabilities.
4.  **Decision Engine Components:**
    *   **Profiler:** Extracts task attributes.
    *   **State Manager:** Aggregates network and device metrics.
    *   **Optimizer:** Runs the APEATO cost function.
    *   **Dispatcher:** Routes the task data to the selected venue.

## 8. Methodology
The simulation follows a rigorous experimental flow:
1.  **Initialization:** Load dataset and initialize network parameters (Bandwidth: 10-100 Mbps, Latency: 10-200ms).
2.  **Task Arrival:** Tasks are dispatched sequentially from the dataset.
3.  **State Evaluation:** For each task `t`, read current `Battery_Level` and `Network_BW`.
4.  **Algorithm Execution:** Run APEATO to select `Target_Node` (Local/Edge/Cloud).
5.  **Performance Calculation:**
    *   *Latency* = Transmission Time + Processing Time + Propagation Delay.
    *   *Energy* = Transmission Power + Idle Power + Processing Power.
6.  **Validation:** Check if `Total_Time <= Deadline`.
7.  **Logging:** Record `Status` (Success/Fail), `Energy_Consumed`, and `Latency` for analysis.

## 9. Expected Outcomes
The proposed APEATO algorithm is expected to demonstrate superior performance compared to baseline methods (Random, Greedy, and Static Offloading):
*   **Energy Efficiency:** A projected **20-30% reduction** in average energy consumption for mobile devices.
*   **Deadline Hit Rate:** A significant increase in the percentage of tasks meeting their deadlines, particularly for high-priority workloads.
*   **Load Balancing:** improved resource utilization across Edge nodes, preventing bottlenecks.
*   **Adaptability:** Robust performance maintenance even under simulated network degradation (e.g., bandwidth drops).

## 10. Conclusion
This project presents a comprehensive study on intelligent task orchestration in Fog-Edge-Cloud environments. By moving beyond classical CPU scheduling and adopting the context-aware **APEATO algorithm**, we address the critical limitations of latency and energy in modern IoT applications. The detailed system architecture and rigorous methodology ensure a valid evaluation, promising a robust solution for next-generation mobile computing systems.

## 11. References
1.  **Bonomi, F., Milito, R., Zhu, J., & Addepalli, S. (2012).** "Fog Computing and Its Role in the Internet of Things." *Proceedings of the First Edition of the MCC Workshop on Mobile Cloud Computing*, ACM, pp. 13-16.
2.  **Mao, Y., You, C., Zhang, J., Huang, K., & Letaief, K. B. (2017).** "A Survey on Mobile Edge Computing: The Communication Perspective." *IEEE Communications Surveys & Tutorials*, 19(4), 2322-2358.
3.  **Silberschatz, A., Galvin, P. B., & Gagne, G. (2018).** *Operating System Concepts*, 10th Edition. Wiley. (For comparison with classical CPU scheduling algorithms).
4.  **Mahmud, R., Kotagiri, R., & Buyya, R. (2018).** "Fog Computing: A Taxonomy, Survey and Future Directions." *Internet of Things*, Elsevier, pp. 103-130.
5.  **Zhang, Y., & Ansari, N. (2020).** "Offloading in Mobile Edge Computing with Energy-Delay Trade-off." *IEEE Internet of Things Journal*, 7(4), 3220-3231.
6.  **Kumar, K., & Lu, Y. H. (2010).** "Cloud Computing for Mobile Users: Can Offloading Computation Save Energy?" *Computer*, 43(4), 51-56.
