import sys
import subprocess

if __name__ == '__main__':
    print("[*] Starting Streamlit server...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard/dashboard.py"])