#!/usr/bin/env python3
"""Validate seeded MTP overlap, rollback, cancellation, and lane reuse over HTTP."""
import argparse
import concurrent.futures
import json
import re
import threading
import time
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument('--base', default='http://127.0.0.1:18082')
parser.add_argument('--tokens', type=int, default=128)
args = parser.parse_args()


def get_json(path):
    with urllib.request.urlopen(args.base + path, timeout=300) as response:
        return json.load(response)


def metrics():
    with urllib.request.urlopen(args.base + '/metrics', timeout=10) as response:
        values = {}
        for line in response.read().decode().splitlines():
            match = re.fullmatch(r'([a-zA-Z_][\w]*(?:\{lane="\d+"\})?) ([0-9]+(?:\.[0-9]+)?)', line)
            if match:
                values[match.group(1)] = float(match.group(2))
        return values


model_info = get_json('/v1/models')['data'][0]
model = model_info['id']
if model_info['mtp_depth'] < 1:
    raise RuntimeError('server does not have MTP enabled')


def request(seed, tokens, barrier=None, stream=False):
    body = {'model': model, 'messages': [{'role': 'user', 'content':
            'Describe a robust bounded work queue and its cancellation tests. ' * 30}],
            'max_tokens': tokens, 'temperature': 0.7, 'seed': seed, 'stream': stream}
    req = urllib.request.Request(args.base + '/v1/chat/completions',
                                 data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    if barrier:
        barrier.wait()
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=300) as response:
        if stream:
            for line in response:
                if not line.startswith(b'data: '):
                    continue
                data = line[6:].strip()
                if data == b'[DONE]':
                    raise RuntimeError('stream ended before cancellation')
                chunk = json.loads(data)
                delta = chunk['choices'][0].get('delta', {}) if chunk.get('choices') else {}
                if delta.get('content') or delta.get('reasoning_content'):
                    return {'closed_after_token': True}
            raise RuntimeError('stream closed before cancellation')
        result = json.load(response)
    return {'seconds': time.monotonic() - start, 'choice': result['choices'][0],
            'usage': result['usage']}


def same(a, b):
    return (a['choice']['message'] == b['choice']['message'] and
            a['choice']['finish_reason'] == b['choice']['finish_reason'] and
            a['usage'] == b['usage'])


before = metrics()
serial = [request(seed, args.tokens) for seed in (11, 29)]
recovery_baseline = request(53, 64)
barrier = threading.Barrier(3)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    futures = [pool.submit(request, seed, args.tokens, barrier) for seed in (11, 29)]
    barrier.wait()
    peak = 0
    while not all(f.done() for f in futures):
        peak = max(peak, int(metrics()['mellum_active_requests']))
        time.sleep(0.01)
    parallel = [f.result() for f in futures]
after_overlap = metrics()

barrier = threading.Barrier(3)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    cancelled = pool.submit(request, 71, 512, barrier, True)
    survivor = pool.submit(request, 83, 1024, barrier)
    barrier.wait()
    cancelled_result = cancelled.result(timeout=300)
    # Wait for the disconnected lane to reset while the survivor stays active.
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        m = metrics()
        if m['mellum_requests_cancelled_total'] > after_overlap['mellum_requests_cancelled_total']:
            break
        time.sleep(0.01)
    else:
        raise RuntimeError('cancelled request did not finish')
    if survivor.done():
        raise RuntimeError('survivor finished before lane reuse could be tested')
    recovered = request(53, 64)
    survivor_result = survivor.result(timeout=300)
final = metrics()
proposed = sum(final.get(f'padra_lane_mtp_proposed_total{{lane="{i}"}}', 0) -
               before.get(f'padra_lane_mtp_proposed_total{{lane="{i}"}}', 0) for i in range(2))
accepted = sum(final.get(f'padra_lane_mtp_accepted_total{{lane="{i}"}}', 0) -
               before.get(f'padra_lane_mtp_accepted_total{{lane="{i}"}}', 0) for i in range(2))
report = {'mtp_depth': model_info['mtp_depth'], 'peak_active': peak,
          'serial_parallel_match': [same(a, b) for a, b in zip(serial, parallel)],
          'cancelled': cancelled_result, 'recovery_matches': same(recovery_baseline, recovered),
          'survivor_completion_tokens': survivor_result['usage']['completion_tokens'],
          'draft_weight_uploads': [final.get(f'padra_lane_mtp_weight_uploads_total{{lane="{i}"}}') for i in range(2)],
          'mtp_proposed_delta': proposed, 'mtp_accepted_delta': accepted,
          'lane_groups': [final.get(f'padra_lane_mtp_groups_total{{lane="{i}"}}') for i in range(2)],
          'active_after': final['mellum_active_requests'],
          'serial_seconds': [r['seconds'] for r in serial],
          'parallel_seconds': [r['seconds'] for r in parallel]}
print(json.dumps(report, indent=2), flush=True)
if not (peak == 2 and all(report['serial_parallel_match']) and report['recovery_matches'] and
        report['survivor_completion_tokens'] > 0 and report['draft_weight_uploads'] == [1, 0] and
        all(report['lane_groups']) and proposed > accepted and report['active_after'] == 0):
    raise SystemExit(1)
