# fog_ui_enhanced.py
# Enhanced Fog Simulator UI with Task Flow Visualization
# Requirements: PyQt5, pyqtgraph, numpy
# Usage: python3 fog_ui_enhanced.py

import sys
import os
import csv
import math
import time
import random
from datetime import datetime
from functools import partial

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QTabWidget, QGroupBox, QComboBox, QSpinBox, QDoubleSpinBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QSplitter, QStatusBar,
    QMessageBox, QFormLayout, QGraphicsOpacityEffect, QScrollArea, QGridLayout
)
from PyQt5.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QPoint, QPropertyAnimation, QEasingCurve,
    QRectF, QSize
)
from PyQt5.QtGui import (
    QPainter, QColor, QPen, QFont, QBrush, QPalette
)
import pyqtgraph as pg
import numpy as np

# Import your APEATO implementation
from APEATO import (
    APEATOAlgorithm, Task as TaskDataclass, SystemState, Priority, Activity
)
try:
    from APEATO import Location
except Exception:
    Location = None

CSV_FILE = 'diverse_tasks.csv'

# Task States
STATE_QUEUED = 'Queued'
STATE_PROCESSING = 'Processing'
STATE_COMPLETED = 'Completed'

# ---------------------------
# Worker Thread (Simulation)
# ---------------------------
class FogSimulationThread(QThread):
    task_signal = pyqtSignal(object, dict)
    finished_signal = pyqtSignal()

    def __init__(self, apeato: APEATOAlgorithm, tasks_list, speed=1.0, parent=None):
        super().__init__(parent)
        self.apeato = apeato
        self.tasks = tasks_list
        self.running = False
        self.paused = False
        self.index = 0
        self.speed = speed
        self.start_time = time.time()

    def run(self):
        self.running = True
        self.paused = False
        
        while self.running and self.index < len(self.tasks):
            if self.paused:
                time.sleep(0.1)
                continue

            task = self.tasks[self.index]

            # Build realistic SystemState
            elapsed = time.time() - self.start_time
            battery = max(5.0, 100.0 - (elapsed * 0.02) % 100)
            cpu_load = min(95.0, max(5.0, 20.0 + (math.sin(elapsed / 20.0) * 30.0) + random.uniform(-5, 5)))
            bandwidth = max(1e6, min(200e6, 30e6 + (math.cos(elapsed / 30.0) * 40e6) + random.uniform(-5e6, 5e6)))
            time_of_day = datetime.now().hour
            activity = random.choice(list(Activity))

            state = SystemState(
                battery=battery,
                cpu_load=cpu_load,
                bandwidth=bandwidth,
                time_of_day=time_of_day,
                activity=activity
            )

            # Get APEATO decision
            try:
                loc = self.apeato.make_decision(task, state)
            except Exception as e:
                loc = None

            # Prepare emit payload
            emit_state = {
                'battery': state.battery,
                'cpu_load': state.cpu_load,
                'bandwidth': state.bandwidth,
                'time': time.time() - self.start_time,
                'activity': activity.name,
                'decision': (loc.name if hasattr(loc, 'name') else str(loc)),
            }

            self.task_signal.emit(task, emit_state)

            # Wait based on speed
            base_wait = 0.8
            wait_time = base_wait / max(0.2, self.speed)
            slept = 0.0
            while self.running and slept < wait_time:
                if self.paused:
                    time.sleep(0.05)
                else:
                    time.sleep(0.05)
                    slept += 0.05
            self.index += 1

        self.finished_signal.emit()

    def stop(self):
        self.running = False

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def set_speed(self, speed):
        self.speed = speed


# ---------------------------
# Task Card Widget
# ---------------------------
class TaskCard(QLabel):
    def __init__(self, task: TaskDataclass, decision_info: dict, parent=None):
        super().__init__(parent)
        self.task = task
        self.decision_info = decision_info
        self.setFixedSize(280, 110)
        self.setup_ui()
        
    def setup_ui(self):
        priority_colors = {
            Priority.HIGH: '#ff6b6b',
            Priority.MEDIUM: '#ffd93d',
            Priority.LOW: '#6bcf7f'
        }
        
        decision = self.decision_info.get('decision', 'UNKNOWN')
        location_colors = {
            'DEVICE': '#3b82f6',
            'EDGE': '#10b981',
            'CLOUD': '#f59e0b'
        }
        
        # Determine location color
        loc_color = location_colors.get('CLOUD', '#f59e0b')
        for key in location_colors:
            if key in decision.upper():
                loc_color = location_colors[key]
                break
        
        priority_color = priority_colors.get(self.task.priority, '#ffd93d')
        
        self.setStyleSheet(f"""
            QLabel {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {loc_color}, stop:1 rgba(30,30,40,220));
                border: 2px solid {priority_color};
                border-radius: 8px;
                padding: 8px;
                color: white;
            }}
        """)
        
        # Create content
        content = f"""
<div style='font-family: Arial; font-size: 11px;'>
    <div style='font-size: 13px; font-weight: bold; margin-bottom: 4px;'>
        Task #{self.task.task_id}
    </div>
    <div style='margin-bottom: 3px;'>
        <span style='color: {priority_color}; font-weight: bold;'>●</span> 
        {self.task.priority.name} Priority
    </div>
    <div style='margin-bottom: 2px;'>Workload: {self.task.workload/1e9:.2f} GC</div>
    <div style='margin-bottom: 2px;'>Data: {self.task.data_size/1e6:.2f} MB</div>
    <div style='margin-bottom: 2px;'>Deadline: {self.task.deadline:.1f}s</div>
    <div style='background: rgba(0,0,0,0.4); padding: 3px; border-radius: 4px; margin-top: 4px; text-align: center; font-weight: bold;'>
        → {decision}
    </div>
</div>
        """
        self.setText(content)


# ---------------------------
# Task Flow Visualization Widget
# ---------------------------
class TaskFlowWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(500)
        
        # Task queues for each state
        self.queued_tasks = []
        self.processing_tasks = []
        self.completed_tasks = []
        
        # Counters
        self.total_queued = 0
        self.total_processing = 0
        self.total_completed = 0
        
        self.setup_ui()
        
        # Animation timer
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self.update_flow)
        self.anim_timer.start(100)
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Header with counters
        header = QHBoxLayout()
        self.counter_queued = QLabel("Queued: 0")
        self.counter_processing = QLabel("Processing: 0")
        self.counter_completed = QLabel("Completed: 0")
        
        for counter in [self.counter_queued, self.counter_processing, self.counter_completed]:
            counter.setStyleSheet("""
                QLabel {
                    background: rgba(50,60,80,200);
                    color: white;
                    padding: 8px 16px;
                    border-radius: 6px;
                    font-size: 13px;
                    font-weight: bold;
                }
            """)
            header.addWidget(counter)
        header.addStretch()
        layout.addLayout(header)
        
        # Main flow area with 3 columns
        flow_layout = QHBoxLayout()
        
        # Queued column
        queued_group = QGroupBox("📥 Queued Tasks")
        queued_group.setStyleSheet("""
            QGroupBox {
                background: rgba(30,35,45,220);
                border: 2px solid #3b82f6;
                border-radius: 8px;
                padding: 10px;
                margin-top: 10px;
                color: white;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        queued_layout = QVBoxLayout()
        self.queued_scroll = QScrollArea()
        self.queued_scroll.setWidgetResizable(True)
        self.queued_container = QWidget()
        self.queued_vlayout = QVBoxLayout(self.queued_container)
        self.queued_vlayout.addStretch()
        self.queued_scroll.setWidget(self.queued_container)
        queued_layout.addWidget(self.queued_scroll)
        queued_group.setLayout(queued_layout)
        
        # Processing column
        processing_group = QGroupBox("⚙️ Processing Tasks")
        processing_group.setStyleSheet("""
            QGroupBox {
                background: rgba(30,35,45,220);
                border: 2px solid #fbbf24;
                border-radius: 8px;
                padding: 10px;
                margin-top: 10px;
                color: white;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        processing_layout = QVBoxLayout()
        self.processing_scroll = QScrollArea()
        self.processing_scroll.setWidgetResizable(True)
        self.processing_container = QWidget()
        self.processing_vlayout = QVBoxLayout(self.processing_container)
        self.processing_vlayout.addStretch()
        self.processing_scroll.setWidget(self.processing_container)
        processing_layout.addWidget(self.processing_scroll)
        processing_group.setLayout(processing_layout)
        
        # Completed column
        completed_group = QGroupBox("✅ Completed Tasks")
        completed_group.setStyleSheet("""
            QGroupBox {
                background: rgba(30,35,45,220);
                border: 2px solid #10b981;
                border-radius: 8px;
                padding: 10px;
                margin-top: 10px;
                color: white;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        completed_layout = QVBoxLayout()
        self.completed_scroll = QScrollArea()
        self.completed_scroll.setWidgetResizable(True)
        self.completed_container = QWidget()
        self.completed_vlayout = QVBoxLayout(self.completed_container)
        self.completed_vlayout.addStretch()
        self.completed_scroll.setWidget(self.completed_container)
        completed_layout.addWidget(self.completed_scroll)
        completed_group.setLayout(completed_layout)
        
        flow_layout.addWidget(queued_group)
        flow_layout.addWidget(processing_group)
        flow_layout.addWidget(completed_group)
        
        layout.addLayout(flow_layout)
        
    def add_task(self, task: TaskDataclass, decision_info: dict):
        """Add new task to queued state"""
        card = TaskCard(task, decision_info)
        self.queued_vlayout.insertWidget(self.queued_vlayout.count() - 1, card)
        self.queued_tasks.append({'card': card, 'task': task, 'time': time.time()})
        self.total_queued += 1
        self.update_counters()
        
        # Auto-move to processing after 1.5 seconds
        QTimer.singleShot(1500, lambda: self.move_to_processing(task.task_id))
        
    def move_to_processing(self, task_id):
        """Move task from queued to processing"""
        for i, item in enumerate(self.queued_tasks):
            if item['task'].task_id == task_id:
                card = item['card']
                task = item['task']
                
                # Remove from queued
                self.queued_vlayout.removeWidget(card)
                self.queued_tasks.pop(i)
                
                # Add to processing with animation
                self.processing_vlayout.insertWidget(self.processing_vlayout.count() - 1, card)
                self.processing_tasks.append({'card': card, 'task': task, 'time': time.time()})
                self.total_processing += 1
                
                # Animate card appearance
                effect = QGraphicsOpacityEffect(card)
                card.setGraphicsEffect(effect)
                anim = QPropertyAnimation(effect, b"opacity")
                anim.setDuration(500)
                anim.setStartValue(0.3)
                anim.setEndValue(1.0)
                anim.start()
                
                self.update_counters()
                
                # Auto-complete after processing time based on workload
                processing_time = int(2000 + (task.workload / 1e9) * 500)
                QTimer.singleShot(processing_time, lambda: self.move_to_completed(task_id))
                break
                
    def move_to_completed(self, task_id):
        """Move task from processing to completed"""
        for i, item in enumerate(self.processing_tasks):
            if item['task'].task_id == task_id:
                card = item['card']
                task = item['task']
                
                # Remove from processing
                self.processing_vlayout.removeWidget(card)
                self.processing_tasks.pop(i)
                
                # Add to completed
                self.completed_vlayout.insertWidget(0, card)  # Insert at top
                self.completed_tasks.append({'card': card, 'task': task, 'time': time.time()})
                self.total_completed += 1
                
                # Fade effect
                effect = QGraphicsOpacityEffect(card)
                card.setGraphicsEffect(effect)
                anim = QPropertyAnimation(effect, b"opacity")
                anim.setDuration(400)
                anim.setStartValue(0.4)
                anim.setEndValue(0.85)
                anim.start()
                
                self.update_counters()
                
                # Keep only recent completed tasks (max 20)
                if len(self.completed_tasks) > 20:
                    oldest = self.completed_tasks.pop(0)
                    self.completed_vlayout.removeWidget(oldest['card'])
                    oldest['card'].deleteLater()
                
                break
                
    def update_counters(self):
        """Update counter displays"""
        self.counter_queued.setText(f"Queued: {len(self.queued_tasks)} (Total: {self.total_queued})")
        self.counter_processing.setText(f"Processing: {len(self.processing_tasks)} (Total: {self.total_processing})")
        self.counter_completed.setText(f"Completed: {len(self.completed_tasks)} (Total: {self.total_completed})")
        
    def update_flow(self):
        """Animation update - could add visual effects here"""
        pass


# ---------------------------
# Decision Breakdown Widget
# ---------------------------
class DecisionBreakdownWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.decision_counts = {'DEVICE': 0, 'EDGE': 0, 'CLOUD': 0}
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        title = QLabel("Decision Distribution")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: white; margin-bottom: 5px;")
        layout.addWidget(title)
        
        # Bars for each decision type
        self.device_bar = self.create_bar("Device", "#3b82f6")
        self.edge_bar = self.create_bar("Edge", "#10b981")
        self.cloud_bar = self.create_bar("Cloud", "#f59e0b")
        
        layout.addWidget(self.device_bar['container'])
        layout.addWidget(self.edge_bar['container'])
        layout.addWidget(self.cloud_bar['container'])
        layout.addStretch()
        
    def create_bar(self, name, color):
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 5, 0, 5)
        
        label = QLabel(name)
        label.setFixedWidth(80)
        label.setStyleSheet("color: white; font-size: 12px;")
        
        bar_container = QWidget()
        bar_container.setFixedHeight(25)
        bar_container.setStyleSheet(f"background: rgba(50,50,60,150); border-radius: 4px;")
        
        bar = QLabel(bar_container)
        bar.setStyleSheet(f"background: {color}; border-radius: 4px;")
        bar.setGeometry(0, 0, 0, 25)
        
        count_label = QLabel("0")
        count_label.setFixedWidth(60)
        count_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        count_label.setStyleSheet("color: white; font-size: 12px; font-weight: bold;")
        
        layout.addWidget(label)
        layout.addWidget(bar_container, 1)
        layout.addWidget(count_label)
        
        return {'container': container, 'bar': bar, 'count': count_label, 'bar_container': bar_container}
        
    def add_decision(self, decision: str):
        """Record a new decision and update bars"""
        decision_upper = decision.upper()
        
        if 'DEVICE' in decision_upper:
            self.decision_counts['DEVICE'] += 1
        elif 'EDGE' in decision_upper:
            self.decision_counts['EDGE'] += 1
        else:
            self.decision_counts['CLOUD'] += 1
            
        self.update_bars()
        
    def update_bars(self):
        """Update bar widths based on counts"""
        total = sum(self.decision_counts.values())
        if total == 0:
            return
            
        for key, bar_dict in [('DEVICE', self.device_bar), ('EDGE', self.edge_bar), ('CLOUD', self.cloud_bar)]:
            count = self.decision_counts[key]
            percentage = count / total
            width = int(bar_dict['bar_container'].width() * percentage)
            
            bar_dict['bar'].setFixedWidth(width)
            bar_dict['count'].setText(f"{count} ({percentage*100:.1f}%)")


# ---------------------------
# Main GUI
# ---------------------------
class FogSimulatorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("APEATO Enhanced - Task Flow Visualizer")
        self.setGeometry(80, 40, 1600, 950)
        self.apeato = APEATOAlgorithm()

        # Load tasks
        self.tasks = self._load_tasks_from_csv(CSV_FILE)
        if not self.tasks:
            QMessageBox.critical(self, "No tasks", f"Could not load tasks from {CSV_FILE}")
            sys.exit(1)

        self.setup_ui()
        
        # Auto-start
        QTimer.singleShot(500, self.start_simulation)
        
    def setup_ui(self):
        # Set dark theme
        self.setStyleSheet("""
            QMainWindow {
                background: #1a1f2e;
            }
            QGroupBox {
                color: white;
                font-weight: bold;
            }
            QLabel {
                color: white;
            }
            QPushButton {
                background: #3b4a6b;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #4a5a7b;
            }
            QPushButton:pressed {
                background: #2a3a5b;
            }
        """)
        
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        
        # Top controls
        ctrl_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("▶ Start")
        self.pause_btn = QPushButton("⏸ Pause")
        self.stop_btn = QPushButton("⏹ Stop")
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.25x", "0.5x", "1x", "2x", "4x"])
        self.speed_combo.setCurrentIndex(2)
        
        for btn in [self.start_btn, self.pause_btn, self.stop_btn]:
            btn.setMinimumWidth(100)
            
        ctrl_layout.addWidget(self.start_btn)
        ctrl_layout.addWidget(self.pause_btn)
        ctrl_layout.addWidget(self.stop_btn)
        ctrl_layout.addWidget(QLabel("Speed:"))
        ctrl_layout.addWidget(self.speed_combo)
        ctrl_layout.addStretch()
        
        # System metrics badges
        self.badge_battery = QLabel("🔋 Battery: 100%")
        self.badge_cpu = QLabel("💻 CPU: 0%")
        self.badge_bw = QLabel("📡 BW: 0 Mbps")
        for badge in [self.badge_battery, self.badge_cpu, self.badge_bw]:
            badge.setStyleSheet("""
                background: rgba(60,70,90,200);
                padding: 8px 12px;
                border-radius: 6px;
                font-weight: bold;
            """)
            ctrl_layout.addWidget(badge)
            
        main_layout.addLayout(ctrl_layout)
        
        # Main content area
        content_splitter = QSplitter(Qt.Horizontal)
        
        # Left: Task flow visualization
        self.task_flow = TaskFlowWidget()
        content_splitter.addWidget(self.task_flow)
        
        # Right: Metrics and decision breakdown
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Decision breakdown
        self.decision_breakdown = DecisionBreakdownWidget()
        right_layout.addWidget(self.decision_breakdown)
        
        # Plots
        pg.setConfigOptions(antialias=True)
        
        self.cpu_plot = pg.PlotWidget(title="Device CPU Load (%)")
        self.cpu_plot.setBackground('#1a1f2e')
        self.cpu_plot.showGrid(x=True, y=True, alpha=0.3)
        self.cpu_curve = self.cpu_plot.plot(pen=pg.mkPen('#3b82f6', width=2))
        self.cpu_data_x = []
        self.cpu_data_y = []
        
        self.energy_plot = pg.PlotWidget(title="Energy Consumption (J)")
        self.energy_plot.setBackground('#1a1f2e')
        self.energy_plot.showGrid(x=True, y=True, alpha=0.3)
        self.energy_curve_dev = self.energy_plot.plot(pen=pg.mkPen('#3b82f6', width=2), name='Device')
        self.energy_curve_edge = self.energy_plot.plot(pen=pg.mkPen('#10b981', width=2), name='Edge')
        self.energy_curve_cloud = self.energy_plot.plot(pen=pg.mkPen('#f59e0b', width=2), name='Cloud')
        self.energy_x = []
        self.energy_dev = []
        self.energy_edge = []
        self.energy_cloud = []
        
        right_layout.addWidget(self.cpu_plot)
        right_layout.addWidget(self.energy_plot)
        
        content_splitter.addWidget(right_panel)
        content_splitter.setSizes([1000, 600])
        
        main_layout.addWidget(content_splitter)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Connect signals
        self.start_btn.clicked.connect(self.start_simulation)
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.stop_btn.clicked.connect(self.stop_simulation)
        self.speed_combo.currentIndexChanged.connect(self._on_speed_change)
        
        # Internal state
        self.sim_thread = None
        
    def _load_tasks_from_csv(self, filename):
        if not os.path.exists(filename):
            return []
        tasks = []
        try:
            with open(filename, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    p = int(float(row.get('priority', 1)))
                    prio = Priority.HIGH if p == 2 else (Priority.MEDIUM if p == 1 else Priority.LOW)
                    t = TaskDataclass(
                        task_id=int(row.get('task_id', 0)),
                        workload=float(row.get('workload', 0.0)),
                        data_size=float(row.get('data_size', 0.0)),
                        result_size=float(row.get('result_size', 0.0)),
                        priority=prio,
                        deadline=float(row.get('deadline', 1.0))
                    )
                    tasks.append(t)
        except Exception as e:
            print("Error loading tasks:", e)
            return []
        return tasks
        
    def start_simulation(self):
        if self.sim_thread and self.sim_thread.isRunning():
            if self.sim_thread.paused:
                self.sim_thread.resume()
                self.status_bar.showMessage("Simulation resumed")
            return
            
        speed = self._selected_speed()
        self.sim_thread = FogSimulationThread(self.apeato, self.tasks, speed=speed)
        self.sim_thread.task_signal.connect(self._on_new_task)
        self.sim_thread.finished_signal.connect(self._on_sim_finished)
        self.sim_thread.start()
        self.status_bar.showMessage("Simulation started")
        
    def _selected_speed(self):
        txt = self.speed_combo.currentText().strip()
        if txt.endswith('x'):
            try:
                return float(txt[:-1])
            except:
                return 1.0
        return 1.0
        
    def toggle_pause(self):
        if not self.sim_thread:
            return
        if self.sim_thread.paused:
            self.sim_thread.resume()
            self.status_bar.showMessage("Resumed")
        else:
            self.sim_thread.pause()
            self.status_bar.showMessage("Paused")
            
    def stop_simulation(self):
        if self.sim_thread:
            self.sim_thread.stop()
            self.sim_thread.wait(2000)
            self.sim_thread = None
            self.status_bar.showMessage("Stopped")
            
    def _on_speed_change(self, idx):
        speed = self._selected_speed()
        if self.sim_thread:
            self.sim_thread.set_speed(speed)
            
    def _on_sim_finished(self):
        self.status_bar.showMessage("Simulation completed!")
        
    def _on_new_task(self, task: TaskDataclass, state: dict):
        # Add to task flow
        self.task_flow.add_task(task, state)
        
        # Update decision breakdown
        decision = state.get('decision', 'CLOUD')
        self.decision_breakdown.add_decision(decision)
        
        # Update metrics
        self._record_metrics(state, task)
        
    def _record_metrics(self, state: dict, task: TaskDataclass):
        t = state.get('time', time.time())
        cpu = state.get('cpu_load', 0.0)
        bw = state.get('bandwidth', 1e6)
        bw_mbps = bw / 1e6
        
        self.cpu_data_x.append(t)
        self.cpu_data_y.append(cpu)
        
        # Energy estimates
        E_dev = self.apeato.calculate_energy_device(task)
        E_edge = self.apeato.calculate_energy_edge(task)
        E_cloud = self.apeato.calculate_energy_cloud(task)
        
        self.energy_x.append(t)
        self.energy_dev.append(E_dev)
        self.energy_edge.append(E_edge)
        self.energy_cloud.append(E_cloud)
        
        # Keep reasonable data size
        max_points = 200
        if len(self.cpu_data_x) > max_points:
            self.cpu_data_x = self.cpu_data_x[-max_points:]
            self.cpu_data_y = self.cpu_data_y[-max_points:]
            self.energy_x = self.energy_x[-max_points:]
            self.energy_dev = self.energy_dev[-max_points:]
            self.energy_edge = self.energy_edge[-max_points:]
            self.energy_cloud = self.energy_cloud[-max_points:]
            
        # Update plots
        if self.cpu_data_x:
            self.cpu_curve.setData(self.cpu_data_x, self.cpu_data_y)
        if self.energy_x:
            self.energy_curve_dev.setData(self.energy_x, self.energy_dev)
            self.energy_curve_edge.setData(self.energy_x, self.energy_edge)
            self.energy_curve_cloud.setData(self.energy_x, self.energy_cloud)
            
        # Update badges
        self.badge_cpu.setText(f"💻 CPU: {cpu:.1f}%")
        self.badge_bw.setText(f"📡 BW: {bw_mbps:.1f} Mbps")
        battery = state.get('battery', 100.0)
        self.badge_battery.setText(f"🔋 Battery: {battery:.1f}%")
        
    def closeEvent(self, event):
        if self.sim_thread and self.sim_thread.isRunning():
            reply = QMessageBox.question(
                self, 'Quit', 
                'Simulation is running. Quit?', 
                QMessageBox.Yes | QMessageBox.No, 
                QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.sim_thread.stop()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


# ---------------------------
# Run Application
# ---------------------------
def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Set dark palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(26, 31, 46))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(30, 35, 50))
    palette.setColor(QPalette.AlternateBase, QColor(26, 31, 46))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(59, 74, 107))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(59, 130, 246))
    palette.setColor(QPalette.Highlight, QColor(59, 130, 246))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)
    
    window = FogSimulatorGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()