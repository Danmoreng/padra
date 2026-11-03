#!/usr/bin/env python3
"""Check serial versus simultaneous seeded requests on a running two-lane server."""
import argparse
import concurrent.futures
import json
import threading
import time
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument('--base', default='http://127.0.0.1:18082')
parser.add_argument('--tokens', type=int, default=128)
args = parser.parse_args()


def get(path):
    with urllib.request.urlopen(args.base + path, timeout=300) as response:
        return json.load(response)


model = get('/v1/models')['data'][0]['id']


def request(seed, barrier=None):
    body = {'model': model, 'messages': [{'role': 'user', 'content':
            'Describe a robust bounded work queue. ' * 30}],
            'max_tokens': args.tokens, 'temperature': 0.7, 'seed': seed}
    req = urllib.request.Request(args.base + '/v1/chat/completions',
                                 data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    if barrier:
        barrier.wait()
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=300) as response:
        data = json.load(response)
    return {'seconds': time.monotonic() - start, 'choice': data['choices'][0],
            'usage': data['usage']}


serial = [request(11), request(29)]
barrier = threading.Barrier(3)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    futures = [pool.submit(request, seed, barrier) for seed in (11, 29)]
    barrier.wait()
    peak = 0
    while not all(f.done() for f in futures):
        with urllib.request.urlopen(args.base + '/metrics', timeout=10) as response:
            for line in response.read().decode().splitlines():
                if line.startswith('mellum_active_requests '):
                    peak = max(peak, int(line.split()[1]))
        time.sleep(0.02)
    parallel = [f.result() for f in futures]

# IDs and timing differ; the deterministic generated messages and finish reasons do not.
matching = all(a['choice']['message'] == b['choice']['message'] and
               a['choice']['finish_reason'] == b['choice']['finish_reason']
               for a, b in zip(serial, parallel))
print(json.dumps({'peak_active': peak, 'seeded_messages_match': matching,
                  'serial_seconds': [x['seconds'] for x in serial],
                  'parallel_seconds': [x['seconds'] for x in parallel],
                  'serial_usage': [x['usage'] for x in serial],
                  'parallel_usage': [x['usage'] for x in parallel]}, indent=2))
if peak != 2 or not matching:
    raise SystemExit(1)
