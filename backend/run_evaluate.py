import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.config import get_settings
from app.ranking.data import DataLoader
from app.ranking.evaluation import run_evaluation


def main():
    settings = get_settings()
    loader = DataLoader(
        perfume_dir=settings.data_perfume_dir,
        organ_dir=settings.data_organ_dir if settings.data_organ_dir.exists() else None,
    )
    summary = run_evaluation(loader, test_ratio=0.2, top_n=10, k_values=[5, 10], seed=42,
                             output_path=settings.project_root / "evaluation_results.json")
    if "error" in summary:
        print(summary["error"])
        sys.exit(1)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
