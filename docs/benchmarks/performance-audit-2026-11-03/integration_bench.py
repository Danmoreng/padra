"""Measure per-request phase rates and whole-wave throughput; stdlib only."""
import argparse
import concurrent.futures
import hashlib
import json
import statistics
import threading
import time
import urllib.request
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--url', default='http://127.0.0.1:18090')
p.add_argument('--output', required=True)
p.add_argument('--rounds', type=int, default=3)
p.add_argument('--clients', type=int, nargs='+', default=[1, 2, 4, 8])
a = p.parse_args()

def get(path):
    with urllib.request.urlopen(a.url + path, timeout=10) as r:
        return r.read()

models = json.loads(get('/v1/models'))
model = models['data'][0]['id']

def request(workload, index, barrier=None):
    repeats, outputs = (384, 1) if workload == 'prefill-heavy' else (64, 256)
    body = {'model': model, 'messages': [{'role': 'user', 'content':
        f'Benchmark task {index}: Review the following code and suggest improvements.\n' +
        'def mean(xs): return sum(xs) / len(xs)\n' * repeats}],
        'max_tokens': outputs, 'temperature': 0.6, 'top_k': 40, 'seed': 42 + index,
        'stream': False}
    req = urllib.request.Request(a.url + '/v1/chat/completions',
        data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    if barrier:
        barrier.wait()
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            headers = {k.lower(): v for k, v in r.headers.items()}
            result = json.load(r)
        elapsed = time.perf_counter() - start
        usage = result['usage']
        prefill_s = float(headers['x-mellum-prefill-ms']) / 1000
        decode_s = float(headers['x-mellum-decode-ms']) / 1000
        completion = result['choices'][0]
        digest = hashlib.sha256(json.dumps(
            {'message': completion['message'], 'finish_reason': completion['finish_reason']},
            sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        return {'index': index, 'seconds': elapsed, 'usage': usage,
                'headers': {k: v for k, v in headers.items() if k.startswith('x-mellum')},
                'prefill_input_tokens_per_second': usage['prompt_tokens'] / prefill_s if prefill_s else None,
                'decode_output_tokens_per_second': (usage['completion_tokens'] - 1) / decode_s
                    if decode_s and usage['completion_tokens'] > 1 else None,
                'output_sha256': digest, 'error': None}
    except Exception as e:
        return {'index': index, 'seconds': time.perf_counter() - start, 'error': str(e)}

report = {'parameters': vars(a), 'models': models, 'warmup': [], 'waves': []}
for workload in ('prefill-heavy', 'decode-heavy'):
    for i in range(2):
        report['warmup'].append(request(workload, 900000 + i))
    for clients in a.clients:
        for repeat in range(a.rounds):
            barrier = threading.Barrier(clients)
            indices = [clients * 10000 + repeat * 100 + i for i in range(clients)]
            observed = []
            stop = threading.Event()
            def observe():
                while not stop.is_set():
                    try:
                        metrics = get('/metrics').decode()
                        for line in metrics.splitlines():
                            if line.startswith('mellum_active_requests '):
                                observed.append(float(line.split()[-1]))
                    except Exception:
                        pass
                    stop.wait(.01)
            observer = threading.Thread(target=observe, daemon=True)
            observer.start()
            start = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=clients) as pool:
                results = list(pool.map(lambda i: request(workload, i, barrier), indices))
            wall = time.perf_counter() - start
            stop.set()
            observer.join()
            valid = all(r['error'] is None for r in results)
            wave = {'workload': workload, 'clients': clients, 'round': repeat,
                    'wall_seconds': wall, 'peak_active_requests': max(observed, default=0),
                    'requests': results}
            if valid:
                decode_rates = [r['decode_output_tokens_per_second'] for r in results
                                if r['decode_output_tokens_per_second'] is not None]
                wave.update({
                    'aggregate_input_tokens_per_wall_second': sum(r['usage']['prompt_tokens'] for r in results) / wall,
                    'aggregate_output_tokens_per_wall_second': sum(r['usage']['completion_tokens'] for r in results) / wall,
                    'median_prefill_input_tokens_per_second': statistics.median(r['prefill_input_tokens_per_second'] for r in results),
                    'median_decode_output_tokens_per_second': statistics.median(decode_rates) if decode_rates else None,
                    'median_request_seconds': statistics.median(r['seconds'] for r in results)})
            report['waves'].append(wave)
            Path(a.output).write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps({k: v for k, v in wave.items() if k != 'requests'}), flush=True)
if any(r['error'] for wave in report['waves'] for r in wave['requests']):
    raise SystemExit('one or more benchmark requests failed')
