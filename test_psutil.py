
import psutil
import time

print("CPU Percent:", psutil.cpu_percent(interval=1))

net_stats = psutil.net_if_stats()
print("\nNet Stats:")
for nic, stats in net_stats.items():
    print(f"{nic}: speed={stats.speed}MB, isup={stats.isup}")

counters1 = psutil.net_io_counters()
time.sleep(1)
counters2 = psutil.net_io_counters()

bytes_sent = counters2.bytes_sent - counters1.bytes_sent
bytes_recv = counters2.bytes_recv - counters1.bytes_recv

print(f"\nThroughput: Sent {bytes_sent} bytes/sec, Recv {bytes_recv} bytes/sec")
print(f"Total: {(bytes_sent + bytes_recv) * 8 / 1e6:.2f} Mbps")
