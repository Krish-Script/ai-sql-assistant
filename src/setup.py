"""
setup.py -- Task 6: Create missing folders for clean project structure
Run once from the project root:
    python src/setup.py
"""

from pathlib import Path

FOLDERS = [
    "data/raw",
    "data/processed",
    "data/golden",
    "src",
    "tests",
    "outputs",
    "docs",
]

def main():
    print("Setting up project folder structure...\n")
    for folder in FOLDERS:
        path = Path(folder)
        if path.exists():
            print(f"  [exists]  {folder}/")
        else:
            path.mkdir(parents=True, exist_ok=True)
            print(f"  [created] {folder}/")

    # Move golden.json to data/golden if it's still in data/processed
    old = Path("data/processed/golden.json")
    new = Path("data/golden/golden.json")
    if old.exists() and not new.exists():
        import shutil
        shutil.copy(old, new)
        print(f"\n  [copied]  golden.json -> data/golden/golden.json")
        print(f"  [note]    Original kept at data/processed/golden.json")

    print("\nDone! Your project structure:")
    print()
    for folder in FOLDERS:
        files = list(Path(folder).glob("*")) if Path(folder).exists() else []
        print(f"  {folder}/")
        for f in sorted(files):
            if f.is_file():
                print(f"    {f.name}")

if __name__ == "__main__":
    main()