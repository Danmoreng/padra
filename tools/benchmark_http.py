#!/usr/bin/env python3
"""Measure one exact-length request against an already running local server.

The request file is sent unchanged. Padra uses /v1/chat/completions; llama.cpp
uses /completion. Prompt-cache reuse and short completions fail validation.
"""
import argparse
import json
from pathlib import Path
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', choices=['padra', 'llama'], required=True)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prompt-tokens', type=int, required=True)
    parser.add_argument('--output-tokens', type=int, default=1000)
    args = parser.parse_args()
    body = json.loads(args.request.read_text())
    endpoint = '/v1/chat/completions' if args.engine == 'padra' else '/completion'
    req = urllib.request.Request(
        args.base_url.rstrip('/') + endpoint, data=json.dumps(body).encode(),
        headers={'Content-Type': 'application/json'})
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=600) as response:
        result = json.load(response)
        headers = dict(response.headers)
    wall = time.monotonic() - started
    if args.engine == 'padra':
        usage = result['usage']
        prompt = usage['prompt_tokens']
        output = usage['completion_tokens']
        cached = usage['prompt_tokens_details']['cached_tokens']
        prefill = float(headers['X-Mellum-Prefill-Ms']) / 1000
        decode = float(headers['X-Mellum-Decode-Ms']) / 1000
        # The first output token is selected from prefill's last logits.
        steps = output - 1
    else:
        timings = result['timings']
        prompt = result['tokens_evaluated']
        output = result['tokens_predicted']
        # llama.cpp reports prompt_n as the tokens actually evaluated.
        cached = prompt - timings['prompt_n']
        prefill = timings['prompt_ms'] / 1000
        decode = timings['predicted_ms'] / 1000
        # predicted_n includes the first output; llama's rate excludes it too.
        steps = output - 1
    measurement = dict(
        prompt_tokens=prompt, output_tokens=output, cached_prompt_tokens=cached,
        decode_forward_steps=steps, prefill_seconds=prefill,
        decode_seconds=decode, wall_seconds=wall,
        prefill_tokens_per_second=prompt / prefill,
        decode_tokens_per_second=steps / decode)
    record = dict(engine=args.engine, base_url=args.base_url, request=body,
                  response=result, headers=headers, measurement=measurement)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(measurement, indent=2), flush=True)
    if (prompt, output, cached) != (args.prompt_tokens, args.output_tokens, 0):
        raise SystemExit('Token counts/cache do not match the requested benchmark; '
                         'raw response saved for inspection.')


if __name__ == '__main__':
    main()
