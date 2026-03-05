import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description="Shelf Helper CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("recommend")
    rec.add_argument("--session-id", type=int, default=None)
    rec.add_argument("--recipe", type=str, default=None)
    rec.add_argument("--top-n", type=int, default=10)
    rec.add_argument("--method", type=str, default="cosine", choices=["cosine", "nn", "gbm", "knn_gbm"])
    rec.add_argument("--json", action="store_true")

    args = parser.parse_args()
    if args.command != "recommend":
        sys.exit(0)

    if args.session_id is None and not (args.recipe or "").strip():
        parser.error("Provide either --session-id or --recipe")

    from app.services.recommend import RecommendService

    svc = RecommendService()
    try:
        if args.session_id is not None:
            ids, scores, explanations = svc.recommend_by_session(args.session_id, top_n=args.top_n, method=args.method)
        else:
            ids, scores, explanations = svc.recommend_by_recipe(args.recipe.strip(), top_n=args.top_n, method=args.method)
    except Exception as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    if args.json:
        out = {
            "perfume_ids": ids,
            "scores": scores,
            "items": [
                {"perfume_id": pid, "score": sc, "explanation": exp}
                for pid, sc, exp in zip(ids, scores, explanations or [])
            ],
        }
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        for i, (pid, sc) in enumerate(zip(ids, scores), 1):
            exp_str = ""
            if explanations and i - 1 < len(explanations) and explanations[i - 1]:
                exp_str = "  " + ", ".join(f"{e['note']}:{e['contribution']}" for e in explanations[i - 1][:3])
            print(f"  {i}. perfume_id={pid}  score={sc:.4f}{exp_str}")


if __name__ == "__main__":
    main()
