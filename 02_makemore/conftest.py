import sys
from pathlib import Path

import pytest

# 資料夾名稱以數字開頭，不能當 package import；讓測試能直接 `import bigram` / `import names_data`
sys.path.insert(0, str(Path(__file__).parent))

from names_data import load_words  # noqa: E402


@pytest.fixture(scope="session")
def words():
    return load_words()


# 標記為 train 的測試會實際訓練模型，預設跳過，加 --run-train 才執行
def pytest_addoption(parser):
    parser.addoption("--run-train", action="store_true", default=False,
                     help="執行會實際訓練模型的測試（標記為 train）")


def pytest_configure(config):
    config.addinivalue_line("markers", "train: 會實際訓練模型，預設跳過；加 --run-train 才執行")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-train", default=False):
        return
    skip = pytest.mark.skip(reason="會實際訓練模型，加 --run-train 才執行")
    for item in items:
        if item.get_closest_marker("train"):
            item.add_marker(skip)
