import sys
import os
import csv
import math
import time
import random
import psutil 
from datetime import datetime
from functools import partial

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QTabWidget, QGroupBox, QComboBox, QSpinBox, QDoubleSpinBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QSplitter, QStatusBar,
    QMessageBox, QFormLayout, QGraphicsOpacityEffect, QScrollArea, QGridLayout,
    QDialog, QProgressBar, QToolTip
)
from PyQt5.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QPoint, QPropertyAnimation, QEasingCurve,
    QRectF, QSize
)
from PyQt5.QtGui import (
    QPainter, QColor, QPen, QFont, QBrush, QPalette, QCursor
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
        self.sim_time = 0.0 # Pause-aware simulation time

        # Initialize Real-time Monitoring
        psutil.cpu_percent(interval=None) # First call to init counter
        self.last_net_io = psutil.net_io_counters()
        self.last_bw_time = time.time()
        self.current_bandwidth = 10e6 # Start with 10 Mbps estimate

    def run(self):
        self.running = True
        self.paused = False
        last_real_time = time.time()
        
        while self.running and self.index < len(self.tasks):
            current_real_time = time.time()
            dt = current_real_time - last_real_time
            last_real_time = current_real_time

            if self.paused:
                time.sleep(0.1)
                continue
            
            # Increment simulation time only if running
            self.sim_time += dt

            task = self.tasks[self.index]

            # Build realistic SystemState
            try:
                battery = psutil.sensors_battery().percent
                is_plugged = psutil.sensors_battery().power_plugged
            except:
                battery = max(5.0, 100.0 - (self.sim_time * 0.02) % 100) # Fallback
                is_plugged = False

            # Use REAL CPU Load
            try:
                cpu_load = psutil.cpu_percent(interval=None) or 10.0
            except:
                cpu_load = 50.0

            # Real Bandwidth Calculation
            try:
                now = time.time()
                bw_dt = now - self.last_bw_time
                # Update bandwidth calc every ~0.5s to get stable readings
                if bw_dt >= 0.5:
                    cur_io = psutil.net_io_counters()
                    # Total bytes (up + down)
                    delta_bytes = (cur_io.bytes_sent + cur_io.bytes_recv) - \
                                  (self.last_net_io.bytes_sent + self.last_net_io.bytes_recv)
                    
                    # Bits per second
                    if delta_bytes < 0: delta_bytes = 0 # Handle wrap-around potentially
                    self.current_bandwidth = (delta_bytes * 8) / bw_dt
                    
                    self.last_net_io = cur_io
                    self.last_bw_time = now
                
                bandwidth = max(100000.0, self.current_bandwidth) # Min 100kbps safety
            except Exception as e:
                print(f"Bandwidth Error: {e}")
                bandwidth = 10e6 # Fallback 10Mbps
            
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
                loc, details = self.apeato.make_decision(task, state)
            except Exception as e:
                loc = None
                details = {}

            # Prepare emit payload
            emit_state = {
                'battery': state.battery,
                'cpu_load': state.cpu_load,
                'bandwidth': state.bandwidth,
                'time': self.sim_time,
                'activity': activity.name,
                'decision': (loc.name if hasattr(loc, 'name') else str(loc)),
                'details': details
            }

            self.task_signal.emit(task, emit_state)

            # Wait based on speed
            base_wait = 0.8
            wait_time = base_wait / max(0.2, self.speed)
            slept = 0.0
            
            # Use 'while' to check paused state frequently during wait
            while self.running and slept < wait_time:
                current_loop_time = time.time()
                loop_dt = current_loop_time - last_real_time
                last_real_time = current_loop_time
                
                if self.paused:
                    time.sleep(0.05)
                    continue
                else:
                    time.sleep(0.05)
                    slept += 0.05
                    self.sim_time += 0.05 # Add wait time to sim_time

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
# Explanation Dialog
# ---------------------------
# ---------------------------
# Animated Explanation Dialog
# ---------------------------
class AnimatedExplanationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("APEATO Logic - Live Explanation")
        self.setFixedSize(700, 600)
        self.setStyleSheet("""
            QDialog {
                background: #1a1f2e;
                color: white;
            }
            QLabel {
                color: #e5e7eb;
                font-family: Segoe UI, Arial;
            }
        """)
        
        # Main Layout
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(20)
        self.layout.setContentsMargins(30, 30, 30, 30)
        
        # Header
        header = QLabel("How APEATO Decides")
        header.setStyleSheet("font-size: 24px; font-weight: bold; color: #8b5cf6;")
        header.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(header)
        
        # Progress Bar
        self.progress = QProgressBar()
        self.progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid #374151;
                border-radius: 5px;
                text-align: center;
                background: #1f2937;
            }
            QProgressBar::chunk {
                background-color: #8b5cf6;
            }
        """)
        self.progress.setRange(0, 100)
        self.layout.addWidget(self.progress)
        
        # Steps Container
        self.steps_container = QWidget()
        self.steps_layout = QVBoxLayout(self.steps_container)
        self.steps_layout.setSpacing(15)
        self.layout.addWidget(self.steps_container)
        
        self.layout.addStretch()
        
        # Close Button
        self.close_btn = QPushButton("Close")
        self.close_btn.setStyleSheet("""
            QPushButton {
                background: #ef4444; color: white; border: none;
                padding: 10px; border-radius: 6px; font-weight: bold;
            }
            QPushButton:hover { background: #dc2626; }
        """)
        self.close_btn.clicked.connect(self.accept)
        self.layout.addWidget(self.close_btn)
        
        # Animation State
        self.step = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_step)
        self.timer.start(1500) # Execute step every 1.5s
        
    def next_step(self):
        self.step += 1
        val = int((self.step / 4) * 100)
        self.progress.setValue(val)
        
        if self.step == 1:
            self.add_step_card(
                "1. Predict State",
                "Using EWMA (Exponential Weighted Moving Average), I predict future Battery, CPU, and Bandwidth.",
                "#3b82f6"
            )
        elif self.step == 2:
            self.add_step_card(
                "2. Weigh Priorities",
                "State + User Activity = Dynamic Weights.\nLow Battery? -> Maximize Energy Savings.\nGaming? -> Maximize Speed.",
                "#f59e0b"
            )
        elif self.step == 3:
            self.add_step_card(
                "3. Compute Costs",
                "I simulate the task on Device, Edge, and Cloud.\nCost = (wE x Energy) + (wL x Latency) + Penalty",
                "#10b981"
            )
        elif self.step == 4:
            self.add_step_card(
                "4. Final Decision",
                "The location with the lowest Total Cost is chosen.\nIf deadlines exist, I ensure they are met!",
                "#8b5cf6"
            )
            self.timer.stop()
            self.close_btn.setText("Close (Explanation Complete)")
            self.close_btn.setStyleSheet("background: #10b981; color: white; padding: 10px; border-radius: 6px; font-weight: bold;")
            
    def add_step_card(self, title, text, color):
        card = QWidget()
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(0, 0, 0, 10) # Minimal margins
        
        # Title
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {color};")
        
        # Text
        text_lbl = QLabel(text)
        text_lbl.setStyleSheet("font-size: 14px; color: #e5e7eb; line-height: 1.4;")
        text_lbl.setWordWrap(True)
        
        card_layout.addWidget(title_lbl)
        card_layout.addWidget(text_lbl)
        
        # Styling: Transparent background, just text
        card.setStyleSheet(f"""
            QWidget {{
                background: transparent;
                border-left: 3px solid {color}; 
                padding-left: 15px;
            }}
        """)
        
        # Add to layout
        self.steps_layout.addWidget(card)
        
        # Animate Fade In Slide
        effect = QGraphicsOpacityEffect(card)
        card.setGraphicsEffect(effect)
        
        self.anim = QPropertyAnimation(effect, b"opacity")
        self.anim.setDuration(800)
        self.anim.setStartValue(0)
        self.anim.setEndValue(1)
        self.anim.start()



# ---------------------------
# Task Card Widget
# ---------------------------
# ---------------------------
# Task Details Dialog
# ---------------------------
# ---------------------------
# Task Details Dialog (Minimalist Text-Only)
# ---------------------------
# ---------------------------
# Task Details Dialog (Natural Language)
# ---------------------------
class TaskDetailsDialog(QDialog):
    def __init__(self, task: TaskDataclass, decision_info: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Task #{task.task_id} Explanation")
        self.setFixedSize(500, 400)
        self.setStyleSheet("""
            QDialog { background: #1a1f2e; color: #e5e7eb; }
            QLabel { font-family: 'Segoe UI', Arial; font-size: 14px; line-height: 1.5; }
            QPushButton {
                background: #374151;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover { background: #4b5563; }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # Title
        title = QLabel(f"Task #{task.task_id} Reprot")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: white; margin-bottom: 15px;")
        layout.addWidget(title)
        
        # Decision Info
        decision = decision_info.get('decision', 'UNKNOWN')
        battery = decision_info.get('battery', 0)
        activity = decision_info.get('activity', 'Unknown')
        
        desc_text = f"""
        This is a <b>{task.priority.name} Priority</b> task carrying {task.data_size/1e6:.2f} MB of data.
        it has a computational workload of {task.workload/1e9:.2f} Giga-Cycles and requires completion within {task.deadline:.1f} seconds.
        <br><br>
        <b>Algorithm Decision:</b><br>
        APEATO selected the <b>{decision}</b> for execution.
        <br><br>
        <b>Context:</b><br>
        At the time of this decision, the device battery was at {battery:.1f}% and the user was engaged in '{activity}' activity.
        The algorithm determined that offloading to {decision} provided the optimal balance of energy savings and latency compliance under these conditions.
        """
        
        lbl = QLabel(desc_text)
        lbl.setWordWrap(True)
        lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(lbl)
        
        layout.addStretch()
        
        # Close
        btn = QPushButton("Close Report")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)

# ---------------------------
# Task Card Widget (Minimal No-Color)
# ---------------------------
class TaskCard(QLabel):
    def __init__(self, task: TaskDataclass, decision_info: dict, parent=None):
        super().__init__(parent)
        self.task = task
        self.decision_info = decision_info
        self.setFixedSize(280, 100)
        self.setCursor(Qt.PointingHandCursor)
        self.setup_ui()
        
    def setup_ui(self):
        decision = self.decision_info.get('decision', 'UNKNOWN')
        
        # Colors for locations
        loc_colors = {
            'DEVICE': '#60a5fa',  # Blue
            'EDGE': '#34d399',    # Green
            'CLOUD': '#f472b6'    # Pink
        }
        
        # Determine color for this decision
        # Default to white/grey if unknown
        text_color = '#e5e7eb'
        for key, color in loc_colors.items():
            if key in decision.upper():
                text_color = color
                break
        
        # Black/White styles only Use simple white border on hover
        self.setStyleSheet("""
            QLabel {
                background: #1f2937;
                border: 1px solid #374151;
                border-radius: 6px;
                padding: 10px;
                color: #e5e7eb;
            }
            QLabel:hover {
                border: 1px solid #9ca3af; 
                background: #283041;
            }
        """)
        
        # Plain text content, no colored dots, BUT colored destination
        content = f"""
<div style='font-family: Arial; font-size: 11px;'>
    <div style='font-size: 13px; font-weight: bold; color: white; margin-bottom: 5px;'>
        Task #{self.task.task_id}
    </div>
    <div style='margin-bottom: 3px; color: #d1d5db;'>
        Priority: {self.task.priority.name}
    </div>
    <div style='color: #9ca3af;'>Workload: {self.task.workload/1e9:.2f} GC | Data: {self.task.data_size/1e6:.2f} MB</div>
    <div style='margin-top: 5px; font-weight: bold; color: #f3f4f6;'>
        ↦ Processed on <span style='color: {text_color};'>{decision}</span>
    </div>
</div>
        """
        self.setText(content)
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            dialog = TaskDetailsDialog(self.task, self.decision_info, self)
            dialog.exec_()


    

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
        queued_group = QGroupBox("Queued Tasks")
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
        processing_group = QGroupBox("Processing Tasks")
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
        completed_group = QGroupBox("Completed Tasks")
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
# Cost Comparison Widget (The "Good" Part)
# ---------------------------
# ---------------------------
# Modern Cost Analysis Widget (Premium UI)
# ---------------------------
class MetricBar(QWidget):
    def __init__(self, label, color, parent=None):
        super().__init__(parent)
        self.value = 0.0
        self.max_val = 1.0
        self.color = color
        self.text_val = "0.00"
        self.label = label
        self.setFixedHeight(35)
        
    def set_data(self, value, max_val):
        self.value = value
        self.max_val = max_val if max_val > 0 else 1.0
        self.text_val = f"{value:.3f}"
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Bg
        rect = self.rect()
        painter.setBrush(QColor(40, 45, 60))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(rect, 4, 4)
        
        # Bar
        pct = min(1.0, self.value / self.max_val)
        bar_width = int(rect.width() * pct)
        bar_rect = QRectF(0, 0, bar_width, rect.height())
        
        gradient = QColor(self.color)
        painter.setBrush(gradient)
        painter.drawRoundedRect(bar_rect, 4, 4)
        
        # Text
        painter.setPen(Qt.white)
        painter.setFont(QFont("Segoe UI", 8))
        painter.drawText(rect.adjusted(5,0,-5,0), Qt.AlignLeft | Qt.AlignVCenter, self.label)
        painter.drawText(rect.adjusted(5,0,-5,0), Qt.AlignRight | Qt.AlignVCenter, self.text_val)


class LocationCard(QWidget):
    def __init__(self, title, color, parent=None):
        super().__init__(parent)
        self.setFixedWidth(140)
        self.title = title
        self.base_color = color
        self.is_winner = False
        
        layout = QVBoxLayout(self)
        layout.setSpacing(5)
        layout.setContentsMargins(5, 10, 5, 10)
        
        # Header
        self.header = QLabel(title)
        self.header.setAlignment(Qt.AlignCenter)
        self.header.setStyleSheet(f"font-weight: bold; font-size: 14px; color: {color};")
        layout.addWidget(self.header)
        
        # Bars
        self.energy_bar = MetricBar("Energy", "#60a5fa")
        self.latency_bar = MetricBar("Latency", "#34d399")
        self.cost_bar = MetricBar("Cost", "#f472b6")
        
        layout.addWidget(self.energy_bar)
        layout.addWidget(self.latency_bar)
        layout.addWidget(self.cost_bar)
        layout.addStretch()
        
    def update_card(self, energy, latency, cost, max_e, max_l, max_c, is_winner):
        self.is_winner = is_winner
        self.energy_bar.set_data(energy, max_e)
        self.latency_bar.set_data(latency, max_l)
        self.cost_bar.set_data(cost, max_c)
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        rect = self.rect()
        
        # Background
        bg_color = QColor(30, 35, 45)
        if self.is_winner:
            bg_color = QColor(40, 50, 70)
            
        painter.setBrush(bg_color)
        
        # Border
        if self.is_winner:
            painter.setPen(QPen(QColor(self.base_color), 2))
        else:
            painter.setPen(QPen(QColor(60, 60, 70), 1))
            
        painter.drawRoundedRect(rect.adjusted(1,1,-1,-1), 8, 8)


class ModernCostWidget(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Real-time Cost Analysis", parent)
        self.setStyleSheet("""
            QGroupBox {
                background: rgba(26,31,46,200);
                border: 1px solid #4b5563;
                border-radius: 8px;
                padding: 10px; 
                margin-top: 20px;
                color: white;
                font-weight: bold;
                font-size: 14px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        
        layout = QVBoxLayout(self)
        
        # Weights Indicator
        weights_container = QWidget()
        w_layout = QHBoxLayout(weights_container)
        w_layout.setContentsMargins(0,0,0,10)
        
        self.w_energy_lbl = QLabel("Energy Priority: 50%")
        self.w_latency_lbl = QLabel("Latency Priority: 50%")
        
        for lbl in [self.w_energy_lbl, self.w_latency_lbl]:
            lbl.setStyleSheet("color: #d1d5db; font-size: 12px;")
            
        w_layout.addWidget(self.w_energy_lbl)
        w_layout.addStretch()
        w_layout.addWidget(self.w_latency_lbl)
        
        layout.addWidget(weights_container)
        
        # Cards Container
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(10)
        
        self.card_dev = LocationCard("DEVICE", "#60a5fa")
        self.card_edge = LocationCard("EDGE", "#34d399")
        self.card_cloud = LocationCard("CLOUD", "#f59e0b")
        
        cards_layout.addWidget(self.card_dev)
        cards_layout.addWidget(self.card_edge)
        cards_layout.addWidget(self.card_cloud)
        
        layout.addLayout(cards_layout)

    def update_data(self, details: dict):
        if not details: return
        
        energies = details.get('energies', {})
        latencies = details.get('latencies', {})
        costs = details.get('costs', {})
        weights = details.get('weights', (0.5, 0.5))
        
        # Update weights labels
        self.w_energy_lbl.setText(f"Energy Priority: {weights[0]*100:.0f}%")
        self.w_latency_lbl.setText(f"Latency Priority: {weights[1]*100:.0f}%")
        
        # Get Max values for normalization
        max_e = max(energies.values()) if energies else 1.0
        max_l = max(latencies.values()) if latencies else 1.0
        max_c = max(costs.values()) if costs else 1.0
        
        # Find winner
        winner_loc = min(costs, key=costs.get)
        
        def get_vals(loc_enum):
            return (
                energies.get(loc_enum, 0.0),
                latencies.get(loc_enum, 0.0),
                costs.get(loc_enum, 0.0),
                loc_enum == winner_loc
            )

        # Assuming Location Enum mapping
        # We need to map our LocationCard to the Enum keys in the dict
        # The dict keys are Enums. Let's find them by name string.
        
        loc_map = {}
        for k in energies.keys():
            loc_map[k.name] = k
            
        if 'DEVICE' in loc_map:
            e, l, c, w = get_vals(loc_map['DEVICE'])
            self.card_dev.update_card(e, l, c, max_e, max_l, max_c, w)
            
        if 'EDGE' in loc_map:
            e, l, c, w = get_vals(loc_map['EDGE'])
            self.card_edge.update_card(e, l, c, max_e, max_l, max_c, w)
            
        if 'CLOUD' in loc_map:
            e, l, c, w = get_vals(loc_map['CLOUD'])
            self.card_cloud.update_card(e, l, c, max_e, max_l, max_c, w)



# ---------------------------
# Main GUI
# ---------------------------
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
        
        # Title with Emoji
        title = QLabel("☁️ APEATO Fog Simulator")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #8b5cf6;")
        ctrl_layout.addWidget(title)
        
        ctrl_layout.addSpacing(20)
        
        # Emojified Buttons
        self.start_btn = QPushButton("▶️ Start")
        self.start_btn.setStyleSheet("background: #10b981;") # Green
        
        self.pause_btn = QPushButton("⏸️ Pause")
        self.pause_btn.setStyleSheet("background: #f59e0b;") # Orange
        
        self.stop_btn = QPushButton("⏹️ Stop")
        self.stop_btn.setStyleSheet("background: #ef4444;") # Red
        
        self.explain_btn = QPushButton("❓ Explain Algorithm")
        self.explain_btn.setStyleSheet("background: #6366f1;") # Indigo
        
        # Speed Combo
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.25x", "0.5x", "1x", "2x", "4x"])
        self.speed_combo.setCurrentIndex(2)
        
        # Add to layout
        ctrl_layout.addWidget(self.start_btn)
        ctrl_layout.addWidget(self.pause_btn)
        ctrl_layout.addWidget(self.stop_btn)
        ctrl_layout.addWidget(self.explain_btn)
        
        ctrl_layout.addWidget(QLabel("⚡ Speed:"))
        ctrl_layout.addWidget(self.speed_combo)
        
        ctrl_layout.addStretch()
        
        # System metrics badges
        self.badge_battery = QLabel("Battery: 100%")
        self.badge_cpu = QLabel("CPU: 0%")
        self.badge_bw = QLabel("BW: 0 Mbps")
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
        
        # Cost widget removed as per user request
        # self.cost_widget = ModernCostWidget()
        # right_layout.addWidget(self.cost_widget)
        
        # Plots
        pg.setConfigOptions(antialias=True)
        
        # CPU Plot (Filled)
        self.cpu_plot = pg.PlotWidget(title="Device CPU Load (%)")
        self.cpu_plot.setBackground('#1a1f2e')
        self.cpu_plot.showGrid(x=True, y=True, alpha=0.3)
        self.cpu_plot.setLabel('left', 'Load', units='%')
        self.cpu_plot.setLabel('bottom', 'Time')
        
        self.cpu_curve = self.cpu_plot.plot(pen=pg.mkPen('#3b82f6', width=2))
        self.cpu_fill = pg.FillBetweenItem(
            self.cpu_curve, 
            pg.PlotCurveItem([], [], pen=None), 
            brush=pg.mkBrush(59, 130, 246, 50) # Blue with alpha
        )
        self.cpu_plot.addItem(self.cpu_fill)
        
        self.cpu_data_x = []
        self.cpu_data_y = []
        
        # Energy Plot (Modernized)
        self.energy_plot = pg.PlotWidget(title="Energy Estimation (J)")
        self.energy_plot.setBackground('#1a1f2e')
        self.energy_plot.showGrid(x=True, y=True, alpha=0.3)
        self.energy_plot.setLabel('left', 'Energy', units='J')
        self.energy_plot.addLegend(offset=(10, 10))
        
        # Curves (Solid Vibrant Colors)
        self.energy_curve_dev = self.energy_plot.plot(
            pen=pg.mkPen('#60a5fa', width=2), name='Device'
        )
        self.energy_curve_edge = self.energy_plot.plot(
            pen=pg.mkPen('#34d399', width=2), name='Edge'
        )
        self.energy_curve_cloud = self.energy_plot.plot(
            pen=pg.mkPen('#f472b6', width=2), name='Cloud'
        )
        
        # Fills (Subtle Glow)
        self.energy_baseline = pg.PlotCurveItem([], [], pen=None) # Hidden baseline
        
        self.energy_fill_dev = pg.FillBetweenItem(
            self.energy_curve_dev, self.energy_baseline, brush=pg.mkBrush(96, 165, 250, 40)
        )
        self.energy_fill_edge = pg.FillBetweenItem(
            self.energy_curve_edge, self.energy_baseline, brush=pg.mkBrush(52, 211, 153, 40)
        )
        self.energy_fill_cloud = pg.FillBetweenItem(
            self.energy_curve_cloud, self.energy_baseline, brush=pg.mkBrush(244, 114, 182, 40)
        )
        
        self.energy_plot.addItem(self.energy_fill_dev)
        self.energy_plot.addItem(self.energy_fill_edge)
        self.energy_plot.addItem(self.energy_fill_cloud)
        
        self.energy_x = []
        self.energy_dev = []
        self.energy_edge = []
        self.energy_cloud = []
        
        # Connect click events
        self.cpu_plot.scene().sigMouseClicked.connect(self.on_cpu_clicked)
        self.energy_plot.scene().sigMouseClicked.connect(self.on_energy_clicked)
        
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
        self.explain_btn.clicked.connect(self.show_explanation)
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
        
    def show_explanation(self):
        dialog = AnimatedExplanationDialog(self)
        dialog.exec_()
        
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
            
    def on_cpu_clicked(self, event):
        if not self.cpu_data_x:
            return
        
        pos = event.scenePos()
        if self.cpu_plot.plotItem.sceneBoundingRect().contains(pos):
            mousePoint = self.cpu_plot.plotItem.vb.mapSceneToView(pos)
            x_val = mousePoint.x()
            
            # Find closest point
            x_arr = np.array(self.cpu_data_x)
            idx = (np.abs(x_arr - x_val)).argmin()
            
            t = self.cpu_data_x[idx]
            cpu = self.cpu_data_y[idx]
            
            # Determine status
            if cpu > 80:
                status = "<span style='color: #ef4444; font-weight: bold;'>Critical Load!</span>"
                desc = "System is overwhelmed. Offloading is highly likely."
            elif cpu > 50:
                status = "<span style='color: #f59e0b; font-weight: bold;'>High Load</span>"
                desc = "Performance impacting ranges. Offloading preferred."
            else:
                status = "<span style='color: #10b981; font-weight: bold;'>Normal Load</span>"
                desc = "System is stable. Local execution is viable."
            
            msg = f"""
            <div style='font-family: Arial; font-size: 13px; padding: 5px; color: black;'>
                <b>Time: {t:.2f}s</b><br>
                CPU Load: <b>{cpu:.1f}%</b><br>
                Status: {status}<br>
                <i>{desc}</i>
            </div>
            """
            QToolTip.showText(QCursor.pos(), msg)

    def on_energy_clicked(self, event):
        if not self.energy_x:
            return
            
        pos = event.scenePos()
        if self.energy_plot.plotItem.sceneBoundingRect().contains(pos):
            mousePoint = self.energy_plot.plotItem.vb.mapSceneToView(pos)
            x_val = mousePoint.x()
            
            # Find closest point
            x_arr = np.array(self.energy_x)
            idx = (np.abs(x_arr - x_val)).argmin()
            
            t = self.energy_x[idx]
            e_dev = self.energy_dev[idx]
            e_edge = self.energy_edge[idx]
            e_cloud = self.energy_cloud[idx]
            
            # Find optimal
            energies = {'Device': e_dev, 'Edge': e_edge, 'Cloud': e_cloud}
            best_loc = min(energies, key=energies.get)
            
            msg = f"""
            <div style='font-family: Arial; font-size: 13px; padding: 5px; color: black;'>
                <b>Time: {t:.2f}s - Energy Analysis</b><hr>
                Device: {e_dev:.4f} J<br>
                Edge: {e_edge:.4f} J<br>
                Cloud: {e_cloud:.4f} J<br>
                <br>
                Most Efficient: <span style='color: #10b981; font-weight: bold;'>{best_loc}</span><br>
                <i>(APEATO likely chose {best_loc} if Latency allowed)</i>
            </div>
            """
            QToolTip.showText(QCursor.pos(), msg)

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
        
        # Update cost widget
        # self.cost_widget.update_data(state.get('details', {}))
        
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
            # Update fill baseline (y=0)
            self.cpu_fill.setCurves(
                self.cpu_curve, 
                pg.PlotCurveItem(self.cpu_data_x, [0]*len(self.cpu_data_x), pen=None)
            )
            
        if self.energy_x:
            self.energy_curve_dev.setData(self.energy_x, self.energy_dev)
            self.energy_curve_edge.setData(self.energy_x, self.energy_edge)
            self.energy_curve_cloud.setData(self.energy_x, self.energy_cloud)
            
            # Update shared baseline for fills
            self.energy_baseline.setData(self.energy_x, [0]*len(self.energy_x))
            
        # Update badges
        self.badge_cpu.setText(f"CPU: {cpu:.1f}%")
        self.badge_bw.setText(f"BW: {bw_mbps:.1f} Mbps")
        battery = state.get('battery', 100.0)
        self.badge_battery.setText(f"Battery: {battery:.1f}%")
        
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