"""Measure completion throughput including queueing, prefill and HTTP overhead."""
import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--url', default='http://127.0.0.1:8080/v1')
parser.add_argument('--max-tokens', type=int, default=256)
parser.add_argument('--prompt-repeats', type=int, default=64)
parser.add_argument('--rounds', type=int, default=3)
parser.add_argument('--concurrency', type=int, nargs='+', default=[1, 2, 4, 8])
parser.add_argument('--output', default='benchmark-results.json')
args = parser.parse_args()
with urllib.request.urlopen(args.url + '/models', timeout=10) as response:
    model = json.load(response)['data'][0]['id']

def request(index):
    payload = {'model': model, 'messages': [{'role': 'user', 'content':
        f'Benchmark task {index}: Review the following code and suggest improvements.\n' +
        'def mean(xs): return sum(xs) / len(xs)\n' * args.prompt_repeats}],
        'max_tokens': args.max_tokens, 'temperature': 0.6, 'stream': False}
    start = time.perf_counter()
    try:
        req = urllib.request.Request(args.url + '/chat/completions',
            data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=900) as response:
            headers = dict(response.headers)
            result = json.load(response)
        usage = result.get('usage', {})
        return {'seconds': time.perf_counter() - start, 'usage': usage,
                'timing_headers': {k: v for k, v in headers.items() if k.lower().startswith('x-mellum')},
                'error': None if 'completion_tokens' in usage else 'missing usage; throughput unavailable'}
    except Exception as exc:
        return {'seconds': time.perf_counter() - start, 'usage': {}, 'error': str(exc)}

report = {'model': model, 'parameters': vars(args), 'warmup': request(-1), 'runs': []}
for concurrency in args.concurrency:
    for repeat in range(args.rounds):
        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            results = list(pool.map(request, [concurrency * 10000 + repeat * 100 + i for i in range(concurrency)]))
        elapsed = time.perf_counter() - start
        valid = all(r['error'] is None for r in results)
        row = {'concurrency': concurrency, 'round': repeat, 'wall_seconds': elapsed,
               'aggregate_output_tokens_per_second': sum(r['usage'].get('completion_tokens', 0) for r in results) / elapsed if valid else None,
               'median_request_seconds': statistics.median(r['seconds'] for r in results), 'requests': results}
        report['runs'].append(row)
        print(json.dumps(row), flush=True)
        Path(args.output).write_text(json.dumps(report, indent=2))
