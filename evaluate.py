"""Rebuild the original sample and measured benchmark; no network required."""
import json
from pathlib import Path
from demand import sample_data, benchmark

if __name__ == '__main__':
    root = Path(__file__).parent / 'data'
    root.mkdir(exist_ok=True)
    frame = sample_data()
    frame.to_csv(root / 'daily_demand.csv', index=False)
    _, metrics, predictions = benchmark(frame)
    (root / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    predictions.to_csv(root / 'holdout_predictions.csv', index=False)
    print(json.dumps(metrics, indent=2))
