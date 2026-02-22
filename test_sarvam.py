"""Quick test to verify Sarvam AI Document Intelligence works with your API key."""
import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
if not SARVAM_API_KEY:
    print("ERROR: SARVAM_API_KEY not found in .env")
    sys.exit(1)

print(f"API Key: {SARVAM_API_KEY[:10]}...{SARVAM_API_KEY[-4:]}")

from sarvamai import SarvamAI

client = SarvamAI(api_subscription_key=SARVAM_API_KEY)

# Use a test PDF - pick the same one
test_pdf = None
for root, dirs, files in os.walk("uploads/documents"):
    for f in files:
        if f.endswith(".pdf"):
            test_pdf = os.path.join(root, f)
            break
    if test_pdf:
        break

if not test_pdf:
    print("No PDF found in uploads/documents/. Provide a path as argument.")
    if len(sys.argv) > 1:
        test_pdf = sys.argv[1]
    else:
        sys.exit(1)

print(f"Testing with: {test_pdf} ({os.path.getsize(test_pdf)} bytes)")

start = time.time()

print("[1/5] Creating job...")
job = client.document_intelligence.create_job(
    language="en-IN",
    output_format="html"
)
print(f"  Job ID: {job.job_id} ({time.time() - start:.1f}s)")

print("[2/5] Uploading file...")
job.upload_file(test_pdf)
print(f"  Uploaded ({time.time() - start:.1f}s)")

print("[3/5] Starting processing...")
job.start()
print(f"  Started ({time.time() - start:.1f}s)")

print("[4/5] Waiting for completion (this may take a few minutes)...")
# Poll manually so we can see progress
import time as _time
for i in range(120):  # Up to 10 minutes
    try:
        status = job.get_status() if hasattr(job, 'get_status') else None
        if status:
            print(f"  Poll {i+1}: state={status.job_state} ({time.time() - start:.1f}s)")
            if status.job_state in ("completed", "COMPLETED", "failed", "FAILED", "error", "ERROR"):
                break
        else:
            # If no get_status, just try wait_until_complete with a note
            print(f"  Polling... ({time.time() - start:.1f}s)")
    except Exception as e:
        print(f"  Status check error: {e}")
    _time.sleep(5)

# Try wait_until_complete
print("  Calling wait_until_complete()...")
try:
    final_status = job.wait_until_complete()
    print(f"  COMPLETED: state={final_status.job_state} ({time.time() - start:.1f}s)")
except Exception as e:
    print(f"  ERROR: {e} ({time.time() - start:.1f}s)")
    sys.exit(1)

print("[5/5] Downloading output...")
output_path = "test_sarvam_output.zip"
job.download_output(output_path)
print(f"  Downloaded to {output_path} ({os.path.getsize(output_path)} bytes)")
print(f"\nTotal time: {time.time() - start:.1f}s")

# Show what's in the ZIP
import zipfile
with zipfile.ZipFile(output_path) as zf:
    print(f"\nZIP contents ({len(zf.namelist())} files):")
    for name in zf.namelist():
        info = zf.getinfo(name)
        print(f"  {name} ({info.file_size} bytes)")

print("\nDone! Sarvam API is working.")
