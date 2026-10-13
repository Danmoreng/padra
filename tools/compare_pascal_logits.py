"""Compare full-vocabulary distributions on identical teacher-forced contexts.

Inputs are the saved tensors from padra-simt-qualify and its frozen corpus.
KL is a distribution distance, not a task-quality score. All metrics use
temperature 1, no truncation, and natural logarithms. No GPU is needed.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def distribution_metrics(reference, candidate, target):
    if reference.shape != candidate.shape or reference.ndim != 1:
        raise ValueError('logit vectors must have identical one-dimensional shapes')
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError('non-finite logits')

    def log_probabilities(logits):
        shifted = logits.astype(np.float64) - float(logits.max())
        return shifted - np.log(np.exp(shifted).sum())

    lp, lq = log_probabilities(reference), log_probabilities(candidate)
    p, q = np.exp(lp), np.exp(lq)
    lm = np.logaddexp(lp, lq) - np.log(2.)
    return {
        'kl_reference_candidate': max(0., float(np.dot(p, lp - lq))),
        'kl_candidate_reference': max(0., float(np.dot(q, lq - lp))),
        'jensen_shannon': max(0., float((np.dot(p, lp - lm) + np.dot(q, lq - lm)) / 2)),
        'delta_nll': float(lp[target] - lq[target]),
        'top1_agreement': int(reference.argmax() == candidate.argmax()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    corpus = json.loads(args.corpus.read_text())
    records, cases, clusters = [], [], {}
    for index, case in enumerate(corpus):
        rows = []
        for offset, target in enumerate(case['targets']):
            relative = Path(f'tensors/request-{index + 1}/position-{len(case["ids"]) - 1 + offset}-logits.bin')
            a = np.fromfile(args.reference / relative, dtype=np.float32)
            b = np.fromfile(args.candidate / relative, dtype=np.float32)
            if a.size != 98304 or b.size != 98304:
                raise ValueError(f'expected full 98304-token vocabulary: {relative}')
            rows.append(distribution_metrics(a, b, target))
        if not rows:
            raise ValueError('empty corpus case')
        records.extend(rows)
        clusters.setdefault(case['source_sha256'], []).extend(rows)
        cases.append({'name': case['name'], 'tokens': len(rows),
                      **{key: float(np.mean([row[key] for row in rows])) for key in rows[0]}})
    if not records:
        raise ValueError('empty corpus')
    cluster_rows = list(clusters.values())
    counts = np.array([len(rows) for rows in cluster_rows])
    # Resample complete source documents, including their repeated contexts.
    indices = np.random.default_rng(20261013).integers(0, len(counts), (10000, len(counts)))
    metrics = {}
    for key in records[0]:
        values = np.array([row[key] for row in records])
        sums = np.array([sum(row[key] for row in rows) for rows in cluster_rows])
        bootstrap = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
        metrics[key] = {'mean': float(values.mean()), 'p50': float(np.quantile(values, .5)),
                        'p95': float(np.quantile(values, .95)), 'p99': float(np.quantile(values, .99)),
                        'max': float(values.max()), 'mean_95_ci': np.quantile(bootstrap, [.025, .975]).tolist()}
    result = {'reference': str(args.reference), 'candidate': str(args.candidate),
              'corpus_sha256': hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
              'tokens': len(records), 'source_documents': len(clusters),
              'temperature': 1, 'units': 'nats, except top1 agreement fraction',
              'bootstrap_seed': 20261013, 'bootstrap_unit': 'whole source document',
              'metrics': metrics, 'cases': cases}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
