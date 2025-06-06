"""Utils for notebooks"""

import queue
import subprocess
import threading
import time
from datetime import datetime

from IPython.display import clear_output

from fed_rag.exceptions import FedRAGError


class ProcessMonitor:
    """A class for launching, managing and monitoring subprocesses within Jupyter Notebooks.

    NOTE: This is intended mainly for launching and managing FL servers and
    client processes.
    """

    def __init__(self) -> None:
        self.processes: dict[str, subprocess.Popen] = {}
        self.log_queues: dict[str, queue.Queue] = {}
        self.log_buffers: dict[str, list[str]] = {}
        self.running = False

    def start_process(self, name: str, command: str) -> None:
        if name in self.processes:
            raise FedRAGError(
                "There already exists a process with the same name."
            )

        proc = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )

        log_queue: queue.Queue = queue.Queue()

        def read_output() -> None:
            """Read the stdout file handle."""
            try:
                if stdout := proc.stdout:
                    for line in iter(stdout.readline, ""):
                        if line:
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            formatted_line = f"[{timestamp}] {line.strip()}"
                            log_queue.put(formatted_line)

            except Exception as e:
                log_queue.put(f"Error reading output: {e}")

        # read logs in a separate background thread
        thread = threading.Thread(target=read_output, daemon=True)
        thread.start()

        self.processes[name] = proc
        self.log_queues[name] = log_queue
        self.log_buffers[name] = []

        print(f"✅ Started {name} (PID: {proc.pid})")

    def get_logs(self, name: str, lines: int = 20) -> str:
        """Get recent logs for a process"""
        if name not in self.log_buffers:
            return f"Process '{name}' not found"

        # Get new logs from queue
        while not self.log_queues[name].empty():
            try:
                self.log_buffers[name].append(
                    self.log_queues[name].get_nowait()
                )
            except queue.Empty:
                break

        # Keep buffer manageable
        if len(self.log_buffers[name]) > 500:
            self.log_buffers[name].pop(0)

        # Return recent logs
        recent_logs = self.log_buffers[name][-lines:]

        return (
            "\n".join(recent_logs)
            if recent_logs
            else f"No output from {name} yet..."
        )

    def is_running(self, name: str) -> bool:
        """Check if process is running"""
        if name not in self.processes:
            return False
        return self.processes[name].poll() is None

    def stop_process(self, name: str) -> None:
        """Stop a process"""
        if name in self.processes:
            proc = self.processes[name]
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

            # clear queues and stopped process
            del self.log_queues[name]
            del self.processes[name]
            print(f"🛑 Stopped {name}")

    def monitor_live(self, name: str, refresh_interval: int = 2) -> None:
        """Live monitoring of a specific process."""
        if name not in self.processes:
            print(f"Process {name} does not exist")
            return

        self.running = True
        print("📊 Starting live monitoring (Ctrl+C to stop)...")

        try:
            while self.running:
                clear_output(wait=True)

                print("🖥️  PROCESS MONITOR")
                print("=" * 60)

                status = "🟢 RUNNING" if self.is_running(name) else "🔴 STOPPED"
                print(f"\n{name} {status}")
                print("-" * 30)
                logs = self.get_logs(name, 15)
                print(logs)

                print(
                    f"\n🔄 Last updated: {datetime.now().strftime('%H:%M:%S')}"
                )
                print("Press Ctrl+C to stop monitoring")

                time.sleep(refresh_interval)

        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped")
            self.running = False

    def stop_all(self) -> None:
        """Stop all processes"""
        self.running = False
        for name in list(self.processes.keys()):
            self.stop_process(name)
        print("🛑 All processes stopped")
