import tkinter as tk
import subprocess

import requests

EXE_PATH = r"app.exe"  # <-- Update this


exe_process = None


def get_pids_on_port(port):
    try:
        output = subprocess.check_output(
            f'netstat -ano | findstr :{port}', shell=True).decode()
        lines = output.strip().split('\n')
        pids = set()
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                pid = parts[-1]
                pids.add(pid)
        return list(pids)
    except subprocess.CalledProcessError:
        return []


def kill_pid(pid):
    subprocess.run(f'taskkill /PID {pid} /F', shell=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def kill_ports():

    url = 'http://127.0.0.1:8090/api/collections/setting/records'
    response = requests.get(url).json()
    kport=response['items'][0]['port']
    PORTS_TO_KILL=[kport,8090]
    for port in PORTS_TO_KILL:
        for pid in get_pids_on_port(port):
            kill_pid(pid)


def start_app():
    global exe_process
    if exe_process is None or exe_process.poll() is not None:
        exe_process = subprocess.Popen(EXE_PATH,creationflags=subprocess.CREATE_NO_WINDOW)
        print("App started.")


def stop_app():
    global exe_process
    kill_ports()
    if exe_process and exe_process.poll() is None:
        exe_process.terminate()
        exe_process.wait()
        exe_process = None
        print("App stopped.")


def reset_app():
    stop_app()
    start_app()


# --- UI ---
root = tk.Tk()
root.title("App Controller")
root.geometry("300x200")

# Bring to front
root.lift()
root.attributes('-topmost', True)
root.after_idle(root.attributes, '-topmost', False)

start_app()
tk.Button(root, text="Start", command=start_app,
          width=20, height=2).pack(pady=10)
tk.Button(root, text="Stop", command=stop_app,
          width=20, height=2).pack(pady=10)
tk.Button(root, text="Reset", command=reset_app,
          width=20, height=2).pack(pady=10)

root.mainloop()
