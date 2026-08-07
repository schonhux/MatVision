from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.evaluation.event_metrics import evaluate_events


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MatVision event predictions")
    parser.add_argument("--truth", required=True, help="JSON file containing ground-truth events")
    parser.add_argument("--predictions", required=True, help="JSON file containing predicted events")
    parser.add_argument("--output", help="Optional metrics JSON output path")
    args = parser.parse_args()

    truth = _read_events(args.truth)
    predictions = _read_events(args.predictions)
    metrics = evaluate_events(truth, predictions)
    rendered = json.dumps(metrics, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")


def _read_events(path: str) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("events", [])
    if not isinstance(payload, list):
        raise TypeError(f"Expected an event list in {path}")
    return payload


if __name__ == "__main__":
    main()
