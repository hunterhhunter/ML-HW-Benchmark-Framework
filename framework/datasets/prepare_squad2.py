import os
import urllib.request
import json


def validate_squad2(path):
    with open(path, "r", encoding="utf-8") as stream:
        payload = json.load(stream)
    articles = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(articles, list) or not any(
        isinstance(article, dict)
        and any(isinstance(paragraph, dict) and paragraph.get("qas") for paragraph in article.get("paragraphs", []))
        for article in articles
    ):
        raise ValueError(f"SQuAD JSON contains no QA questions: {path}")

def prepare_squad2():
    base_dir = os.path.join("datasets", "squad2")
    os.makedirs(base_dir, exist_ok=True)
    
    url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
    target_path = os.path.join(base_dir, "val.json")
    partial_path = target_path + ".part"
    
    print(f"[*] Downloading SQuAD 2.0 (dev) to {target_path}...")
    try:
        urllib.request.urlretrieve(url, partial_path)
        validate_squad2(partial_path)
        os.replace(partial_path, target_path)
        print("[+] SQuAD 2.0 Download complete!\n")
    except Exception as e:
        if os.path.exists(partial_path):
            os.remove(partial_path)
        print(f"[!] Failed to download SQuAD 2.0: {e}\n")
        raise

if __name__ == "__main__":
    prepare_squad2()
