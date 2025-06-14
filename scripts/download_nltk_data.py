# scripts/download_nltk_data.py
import nltk
from pathlib import Path

# Define the target directory for NLTK data right inside your project
PROJECT_ROOT = Path(__file__).parent.parent
NLTK_DATA_DIR = PROJECT_ROOT / "nltk_data"

# Create the directory if it doesn't exist
if not NLTK_DATA_DIR.exists():
    print(f"Creating NLTK data directory at: {NLTK_DATA_DIR}")
    NLTK_DATA_DIR.mkdir()

# --- THIS IS THE FIX ---
# Add 'punkt_tab' to the list of packages to download
packages = ['punkt', 'stopwords', 'punkt_tab']
# --- END FIX ---

print(f"Downloading NLTK packages to: {NLTK_DATA_DIR}")
for package in packages:
    try:
        print(f"--- Downloading '{package}' ---")
        nltk.download(package, download_dir=str(NLTK_DATA_DIR))
        print(f"--- Successfully downloaded '{package}' ---\n")
    except Exception as e:
        print(f"Error downloading {package}: {e}")

print("NLTK data setup is complete.")