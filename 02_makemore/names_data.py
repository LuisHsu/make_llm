"""makemore 的資料：data/names.txt（Karpathy makemore repo 的名字清單）。

下載：uv run python 02_makemore/names_data.py
"""

import hashlib
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
SHA256 = "0a30b5557f192f32ab962680889aac5f6fda0f4cecf40a6d0b5694f58ea8cc4d"
NAMES_PATH = Path(__file__).resolve().parent.parent / "data" / "names.txt"


def download(path: Path = NAMES_PATH) -> Path:
    """下載 names.txt 並核對 sha256；檔案已存在且內容正確時不重新下載。"""
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == SHA256:
        return path
    with urllib.request.urlopen(URL) as resp:
        raw = resp.read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SHA256:
        raise RuntimeError(f"names.txt 的 sha256 不符：{digest}（預期 {SHA256}）")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path


def load_words(path: Path = NAMES_PATH) -> list[str]:
    """每行一個名字，全部是小寫 a–z。"""
    if not path.exists():
        raise FileNotFoundError(f"{path} 不存在，先執行：uv run python 02_makemore/names_data.py")
    return path.read_text().splitlines()


if __name__ == "__main__":
    p = download()
    print(f"{p}：{len(load_words(p))} 個名字")
