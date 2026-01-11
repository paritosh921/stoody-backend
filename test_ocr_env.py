import os
import sys

# Add current directory to path
sys.path.append(os.getcwd())

from dotenv import load_dotenv
load_dotenv()

try:
    import fitz
    print("✅ PyMuPDF (fitz) imported successfully")
    print(f"PyMuPDF version: {fitz.__doc__}")
except ImportError as e:
    print(f"❌ Failed to import fitz: {e}")

api_key = os.getenv("MISTRAL_API_KEY")
if api_key:
    print(f"✅ MISTRAL_API_KEY is set: {api_key[:5]}...")
else:
    print("❌ MISTRAL_API_KEY is NOT set")
