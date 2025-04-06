import subprocess
import time
import os
import signal
import sys

def start_flask():
    return subprocess.Popen(['python', 'app.py'])

def start_ngrok():
    return subprocess.Popen(['ngrok', 'http', '5000'])

def main():
    flask_process = start_flask()
    time.sleep(5)  # Wait for Flask to start
    ngrok_process = start_ngrok()

    try:
        while True:
            time.sleep(1)
            if flask_process.poll() is not None:
                print("Flask stopped, restarting...")
                flask_process = start_flask()
            if ngrok_process.poll() is not None:
                print("Ngrok stopped, restarting...")
                ngrok_process = start_ngrok()
    except KeyboardInterrupt:
        print("Shutting down...")
        flask_process.terminate()
        ngrok_process.terminate()
        sys.exit(0)

if __name__ == "__main__":
    main() 