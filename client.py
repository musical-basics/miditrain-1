import os
import time
import requests

# The URL of your local FastAPI server
API_URL = "http://localhost:8000/clean-midi"
MESSY_DIR = "./messy_midi_folder" # Folder containing your test files
OUTPUT_DIR = "./output_midi_folder"

os.makedirs(OUTPUT_DIR, exist_ok=True)
files = [f for f in os.listdir(MESSY_DIR) if f.endswith(".mid")]

if len(files) == 0:
    print(f"No .mid files found in {MESSY_DIR}. Drop some MIDI files there and try again!")
    exit(0)

print(f"Found {len(files)} files to process. Piping to API...\n")

for idx, filename in enumerate(files, start=1):
    file_path = os.path.join(MESSY_DIR, filename)
    
    start_time = time.time()
    
    # Send the file to the API
    with open(file_path, "rb") as f:
        response = requests.post(API_URL, files={"file": f})
        
    elapsed_ms = int((time.time() - start_time) * 1000)
    
    # Replicate the screenshot's terminal output formatting
    status_code = response.status_code
    check_mark = "✓" if status_code == 200 else "✗"
    
    if status_code == 200:
        # Save the returned clean MIDI
        output_path = os.path.join(OUTPUT_DIR, f"clean_{filename}")
        with open(output_path, "wb") as f:
            f.write(response.content)
            
        print(f"{idx}/{len(files)} {status_code} {check_mark} {elapsed_ms}ms \"{filename[:30]:<30}\" -> cleaned successfully")
    else:
         print(f"{idx}/{len(files)} {status_code} {check_mark} {elapsed_ms}ms \"{filename[:30]:<30}\" -> API ERROR: {response.text}")
