#!/usr/bin/env python3
"""Summarize paired exact-length Padra/llama.cpp HTTP benchmark records."""
import argparse
import json
import random
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--repetitions', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.repetitions < 5:
        parser.error('at least five paired warm repetitions are required')
    rng = random.Random(20261013)
    historical = {2000: (2.244696, 17.473134), 4000: (4.476068, 17.905969)}
    results = []
    for length in (2000, 4000):
        records = {}
        for engine in ('padra', 'llama'):
            records[engine] = []
            for repetition in range(args.repetitions):
                record = json.loads((args.directory / f'{engine}-{repetition}-{length}.json').read_text())
                measurement = record['measurement']
                expected = dict(prompt_tokens=length, output_tokens=1000,
                                cached_prompt_tokens=0, decode_forward_steps=999)
                if record['engine'] != engine or any(measurement[k] != v for k, v in expected.items()):
                    raise ValueError(f'invalid {engine} repetition {repetition} length {length}')
                records[engine].append(measurement)
        for phase_index, phase in enumerate(('prefill', 'decode')):
            values = {engine: [r[f'{phase}_seconds'] for r in records[engine]] for engine in records}
            speedups = [b / a for a, b in zip(values['padra'], values['llama'])]
            bootstrap = sorted(statistics.median(rng.choices(speedups, k=len(speedups))) for _ in range(10000))
            interval = [bootstrap[249], bootstrap[9749]]
            medians = {engine: statistics.median(v) for engine, v in values.items()}
            count = length if phase == 'prefill' else 999
            results.append(dict(input_tokens=length, phase=phase, seconds=values,
                                median_seconds=medians,
                                min_max_seconds={engine: [min(v), max(v)] for engine, v in values.items()},
                                median_tokens_per_second={engine: count / v for engine, v in medians.items()},
                                paired_speedups=speedups, median_paired_speedup=statistics.median(speedups),
                                paired_speedup_bootstrap_95_percent=interval,
                                historical_llama_target_seconds=historical[length][phase_index],
                                meets_historical_target=medians['padra'] <= historical[length][phase_index],
                                meets_contemporary_target=medians['padra'] <= medians['llama'] and interval[0] >= 1))
    report = dict(repetitions=args.repetitions, bootstrap_seed=20261013, bootstrap_samples=10000,
                  speedup_definition='llama_seconds / padra_seconds; greater than one favors Padra',
                  all_four_targets_met=all(r['meets_historical_target'] and r['meets_contemporary_target'] for r in results),
                  profiles=results)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    for r in results:
        print(r['input_tokens'], r['phase'], r['median_tokens_per_second'],
              'paired speedup CI', r['paired_speedup_bootstrap_95_percent'])


if __name__ == '__main__':
    main()
