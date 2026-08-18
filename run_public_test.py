import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
INFERENCE_SCRIPT = PROJECT_ROOT / "src" / "inference.py"
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "public-official.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "submission.json"

def main():
    input_file = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_INPUT)
    output_file = sys.argv[2] if len(sys.argv) > 2 else str(DEFAULT_OUTPUT)

    print("=" * 60)
    print("CHAY THUC NGHIEM SUY LUAN (INFERENCE) TREN BO PUBLIC TEST")
    print(f"File cau hoi dau vao : {input_file}")
    print(f"File ket qua dau ra   : {output_file}")
    print("=" * 60)

    cmd = [
        sys.executable,
        str(INFERENCE_SCRIPT),
        "--input", str(input_file),
        "--output", str(output_file)
    ]

    try:
        result = subprocess.run(cmd, check=True)
        if result.returncode == 0:
            print("\nDa sinh file ket qua thanh cong!")
            print(f"Vi tri file submission: {os.path.abspath(output_file)}")
    except subprocess.CalledProcessError as e:
        print(f"\nCo loi xay ra trong qua trinh thuc thi: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
