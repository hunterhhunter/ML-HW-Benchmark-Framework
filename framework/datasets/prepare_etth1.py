import os
import urllib.request
import argparse
import csv


def validate_etth1(path):
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        header = next(reader, [])
        if "date" not in header or "OT" not in header:
            raise ValueError(f"ETTh1 CSV is missing required columns: {path}")
        rows = sum(1 for _ in reader)
    if rows < 12116:
        raise ValueError(f"ETTh1 CSV has too few rows: {rows}")

def download_etth1(output_dir="datasets/etth1"):
    os.makedirs(output_dir, exist_ok=True)
    url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
    dest_path = os.path.join(output_dir, "ETTh1.csv")
    
    if os.path.exists(dest_path):
        try:
            validate_etth1(dest_path)
        except (OSError, ValueError) as exc:
            print(f"[*] Existing ETTh1 dataset is invalid ({exc}); downloading again.")
        else:
            print(f"[*] ETTh1 dataset already exists at {dest_path}")
            return
        
    print(f"[*] Downloading ETTh1 dataset from {url}...")
    partial_path = dest_path + ".part"
    try:
        urllib.request.urlretrieve(url, partial_path)
        validate_etth1(partial_path)
        os.replace(partial_path, dest_path)
        print(f"[+] Download complete! Saved to {dest_path}")
    except Exception as e:
        if os.path.exists(partial_path):
            os.remove(partial_path)
        print(f"[Error] Failed to download ETTh1: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=os.path.join(os.path.dirname(__file__), "etth1"))
    args = parser.parse_args()
    
    download_etth1(args.output_dir)
