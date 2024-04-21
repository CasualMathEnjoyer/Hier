import psutil
import time

def log_resource_utilization():
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_percent = psutil.virtual_memory().percent
    print(f"CPU Utilization: {cpu_percent}% | Memory Utilization: {memory_percent}%")

if __name__ == "__main__":
    while True:
        log_resource_utilization()
        time.sleep(2)  # Log every 10 seconds