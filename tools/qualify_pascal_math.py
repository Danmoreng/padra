"""Compare saved teacher-forced logits on a frozen native-token corpus.

Each corpus case contains input IDs and target IDs; tokenization is external.
Run both binaries with identical tokens, and bootstrap whole source documents
(not treating repeated cases or correlated tokens as independent). This measures preservation against the old
engine, not absolute quality against original unquantized model weights.
"""
import argparse
import hashlib
import json
import pathlib
import subprocess

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(directory, case, request_index=0):
    records = []
    first_position = len(case['ids']) - 1
    for offset, target in enumerate(case['targets']):
        path = directory / f'tensors/request-{request_index + 1}' / f'position-{first_position + offset}-logits.bin'
        logits = np.fromfile(path, dtype=np.float32).astype(np.float64)
        if logits.size != 98304 or not np.isfinite(logits).all():
            raise ValueError(f'invalid logits: {path}')
        maximum = logits.max()
        nll = maximum + np.log(np.exp(logits - maximum).sum()) - logits[target]
        order = np.argpartition(logits, -5)[-5:]
        records.append({'nll': float(nll), 'top1': int(logits.argmax()), 'top5': sorted(map(int, order))})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=pathlib.Path, required=True)
    parser.add_argument('--reference-binary', type=pathlib.Path, required=True)
    parser.add_argument('--candidate-binary', type=pathlib.Path, required=True)
    parser.add_argument('--artifact', required=True)
    parser.add_argument('--tokenizer-sha256', required=True)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    identity = {'corpus_sha256': sha(args.corpus), 'reference_sha256': sha(args.reference_binary),
                'candidate_sha256': sha(args.candidate_binary), 'nll_limit': .01,
                'bootstrap_unit': 'whole source document', 'bootstrap_seed': 20261013}
    manifest = args.output / 'identity.json'
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError('output directory belongs to different binaries/corpus')
    manifest.write_text(json.dumps(identity, indent=2))
    corpus = json.loads(args.corpus.read_text())
    # Each engine loads the artifact once. Per-request teacher sequences are
    # explicit offline options; the HTTP runtime cannot select this mode.
    for label, binary in [('reference', args.reference_binary), ('candidate', args.candidate_binary)]:
        if all((args.output / f'{index:02}-{label}-scores.json').exists() for index in range(len(corpus))):
            continue
        config = {'capacity': 8192, 'chunk': 256, 'diagnostics': True, 'layer': 0,
                  'positions': [8191], 'reference_math': label == 'reference',
                  'teacher_sequences': [case['targets'][:-1] for case in corpus],
                  'requests': [{'ids': case['ids'], 'generate': len(case['targets'])} for case in corpus]}
        config_file = args.output / f'{label}.json'
        config_file.write_text(json.dumps(config))
        directory = args.output / label
        print('RUN', label, len(corpus), 'cases', flush=True)
        with (args.output / f'{label}.log').open('w') as log:
            subprocess.run([str(binary), args.artifact, args.tokenizer_sha256,
                            str(config_file), str(directory)], stdout=log, stderr=subprocess.STDOUT, check=True)
        for index, case in enumerate(corpus):
            (args.output / f'{index:02}-{label}-scores.json').write_text(json.dumps(evaluate(directory, case, index)))
    cases = []
    for index, case in enumerate(corpus):
        if len(case['targets']) < 2:
            raise ValueError('teacher replay needs at least two target tokens')
        measurements = {}
        for label in ['reference', 'candidate']:
            summary = args.output / f'{index:02}-{label}-scores.json'
            measurements[label] = json.loads(summary.read_text())
        a, b = measurements['reference'], measurements['candidate']
        cases.append({'name': case['name'], 'source_sha256': case['source_sha256'], 'input_tokens': len(case['ids']), 'scored_tokens': len(a),
                      'reference_nll': float(np.mean([v['nll'] for v in a])),
                      'candidate_nll': float(np.mean([v['nll'] for v in b])),
                      'delta_nll': float(np.mean([y['nll'] - x['nll'] for x, y in zip(a, b)])),
                      'top1_agreement': float(np.mean([x['top1'] == y['top1'] for x, y in zip(a, b)])),
                      'top5_overlap': float(np.mean([len(set(x['top5']) & set(y['top5'])) / 5 for x, y in zip(a, b)]))})
        (args.output / 'cases.json').write_text(json.dumps(cases, indent=2))
    delta = np.array([case['delta_nll'] for case in cases])
    weights = np.array([case['scored_tokens'] for case in cases])
    clusters = {}
    for case in cases:
        weighted_sum, count = clusters.get(case['source_sha256'], (0., 0))
        clusters[case['source_sha256']] = (weighted_sum + case['delta_nll'] * case['scored_tokens'], count + case['scored_tokens'])
    cluster_sums, cluster_counts = np.array(list(clusters.values())).T
    indices = np.random.default_rng(20261013).integers(0, len(clusters), (10000, len(clusters)))
    samples = np.sum(cluster_sums[indices], axis=1) / np.sum(cluster_counts[indices], axis=1)
    interval = np.quantile(samples, [.025, .975]).tolist()
    result = {'identity': identity, 'cases': cases, 'scored_tokens': int(weights.sum()),
              'delta_nll': float(np.average(delta, weights=weights)), 'delta_nll_95_ci': interval,
              'nll_preservation_passed': interval[1] <= .01,
              'limitations': 'Frozen engineering corpus; old quantized engine is the reference. '
                             'This does not establish original-model equivalence or broad task success.'}
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ('identity', 'cases')}, indent=2))
    if not result['nll_preservation_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
