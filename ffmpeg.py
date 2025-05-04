# ffmpeg.py
import subprocess

result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
print(result.stdout)
