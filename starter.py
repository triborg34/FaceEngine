import json
import logging
import subprocess
import sys
import tkinter as tk
from tkinter import messagebox

import requests

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s'
)

EXE_PATH = r"app.exe"
CONFIG_FILE = "hostname.json"
POCKETBASE_PORT = 8091


def read_port():
    try:
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f).get('port', 8000)
    except (FileNotFoundError, json.JSONDecodeError):
        return 8000


def get_pids_on_port(port):
    try:
        output = subprocess.check_output(
            f'netstat -ano | findstr :{port}', shell=True,
            stderr=subprocess.DEVNULL
        ).decode()
        pids = set()
        for line in output.strip().split('\n'):
            parts = line.strip().split()
            if len(parts) >= 5:
                pids.add(parts[-1])
        return list(pids)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []


def kill_pid(pid):
    subprocess.run(
        f'taskkill /PID {pid} /F', shell=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def kill_ports():
    ports_to_kill = [read_port(), POCKETBASE_PORT]
    for port in ports_to_kill:
        for pid in get_pids_on_port(port):
            kill_pid(pid)
            logging.info(f"Killed PID {pid} on port {port}")


class AppController:
    def __init__(self, root):
        self.root = root
        self.root.title("App Controller")
        self.root.geometry("300x200")
        self.exe_process = None
        self._setup_ui()
        self.start_app()

    def _setup_ui(self):
        tk.Button(self.root, text="Start", command=self.start_app,
                  width=20, height=2).pack(pady=10)
        tk.Button(self.root, text="Stop", command=self.stop_app,
                  width=20, height=2).pack(pady=10)
        tk.Button(self.root, text="Reset", command=self.reset_app,
                  width=20, height=2).pack(pady=10)

    def start_app(self):
        if self.exe_process is None or self.exe_process.poll() is not None:
            try:
                self.exe_process = subprocess.Popen(
                    EXE_PATH, creationflags=subprocess.CREATE_NO_WINDOW
                )
                logging.info(f"App started (PID: {self.exe_process.pid})")
            except FileNotFoundError:
                messagebox.showerror("Error", f"Executable not found: {EXE_PATH}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to start app: {e}")

    def stop_app(self):
        kill_ports()
        if self.exe_process and self.exe_process.poll() is None:
            self.exe_process.terminate()
            try:
                self.exe_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.exe_process.kill()
            self.exe_process = None
            logging.info("App stopped")

    def reset_app(self):
        self.stop_app()
        self.start_app()


def main():
    root = tk.Tk()
    root.lift()
    root.attributes('-topmost', True)
    root.after_idle(root.attributes, '-topmost', False)
    AppController(root)
    root.mainloop()


if __name__ == "__main__":
    main()
