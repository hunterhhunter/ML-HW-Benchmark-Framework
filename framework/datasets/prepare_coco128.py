import os
import urllib.request
import zipfile
from pathlib import Path


def validate_coco128(dataset_dir):
    root = Path(dataset_dir)
    image_dir = root / "images" / "train2017"
    label_dir = root / "labels" / "train2017"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"COCO128 images or labels are missing under {root}")
    images = [path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if len(images) != 128:
        raise ValueError(f"COCO128 expected 128 images, found {len(images)}")
    image_stems = {path.stem for path in images}
    labels = list(label_dir.glob("*.txt"))
    if not any(path.stat().st_size > 0 for path in labels):
        raise ValueError(f"COCO128 has no nonempty label files under {label_dir}")
    if any(path.stem not in image_stems for path in labels):
        raise ValueError(f"COCO128 has label files without images under {label_dir}")

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    datasets_dir = os.path.join(project_root, "datasets")
    os.makedirs(datasets_dir, exist_ok=True)
    
    url = "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip"
    zip_path = os.path.join(datasets_dir, "coco128.zip")
    partial_path = zip_path + ".part"
    
    print(f"[*] Downloading COCO128 from {url} to {datasets_dir}...")
    
    def reporthook(block_num, block_size, total_size):
        fetched = block_num * block_size
        if total_size > 0:
            percent = min(100, fetched * 100 // total_size)
            print(f"\rDownloading... {percent}% ({fetched/(1024*1024):.1f}MB / {total_size/(1024*1024):.1f}MB)", end="")

    try:
        urllib.request.urlretrieve(url, partial_path, reporthook)
        os.replace(partial_path, zip_path)
        print("\n[+] Download complete!")
    except Exception as e:
        if os.path.exists(partial_path):
            os.remove(partial_path)
        print(f"[!] 다운로드 실패: {e}")
        raise
    
    print("[*] Extracting zip file...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # ultralytics coco128.zip 에는 자기 자신의 루트 폴더('coco128')가 내장되어 있습니다.
            zip_ref.extractall(datasets_dir)
    except Exception as e:
        print(f"[!] 압축 해제 실패: {e}")
        raise

    validate_coco128(os.path.join(datasets_dir, "coco128"))
    print(f"[+] COCO128 dataset is ready at {os.path.join(datasets_dir, 'coco128')}")

if __name__ == "__main__":
    main()
