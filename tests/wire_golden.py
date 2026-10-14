"""Exact structural wire fixtures; normalize only generated IDs and timestamps."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import urllib.request
import urllib.error


def collect(fixture):
    process = subprocess.Popen([fixture], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        port = json.loads(process.stdout.readline())['port']
        base = f'http://127.0.0.1:{port}'
        for attempt in range(100):
            try:
                urllib.request.urlopen(base + '/healthz', timeout=1).close()
                break
            except OSError:
                time.sleep(.01)
        cases = {}
        for route in ('chat/completions', 'responses'):
            for prompt in ('hello', 'reasoning-text', 'tool-valid', 'tool-reverse',
                           'tool-mixed', 'tool-malformed', 'utf8-stop', 'partial-tool'):
                for stream in (False, True):
                    response_api = route == 'responses'
                    budget = 54 if prompt == 'partial-tool' else 256
                    body = {'model': 'mellum21-exl3-k4-research', 'stream': stream}
                    if response_api:
                        body.update(input='tool-valid' if prompt == 'partial-tool' else prompt,
                                    max_output_tokens=budget)
                    else:
                        body.update(messages=[{'role': 'user', 'content':
                                               'tool-valid' if prompt == 'partial-tool' else prompt}],
                                    max_completion_tokens=budget)
                        if stream:
                            body['stream_options'] = {'include_usage': True}
                    if 'tool' in prompt:
                        tool = {'name': 'lookup', 'parameters': {'type': 'object'}}
                        body['tools'] = [dict(type='function', **tool) if response_api
                                         else {'type': 'function', 'function': tool}]
                    if prompt == 'utf8-stop':
                        body['stop'] = ['STOP']
                    request = urllib.request.Request(base + '/v1/' + route,
                                                     json.dumps(body).encode(),
                                                     {'Content-Type': 'application/json'})
                    try:
                        result = urllib.request.urlopen(request, timeout=5)
                    except urllib.error.HTTPError as error:
                        result = error
                    raw = result.read().decode()
                    if stream and result.status == 200:
                        value = []
                        for record in raw.strip().split('\n\n'):
                            lines = record.splitlines()
                            data = '\n'.join(line[6:] for line in lines if line.startswith('data: '))
                            value.append({'event': next((line[7:] for line in lines
                                                         if line.startswith('event: ')), None),
                                          'data': '[DONE]' if data == '[DONE]' else json.loads(data)})
                    else:
                        value = json.loads(raw)
                    identities = {}
                    def normalize(obj):
                        if isinstance(obj, list):
                            return [normalize(v) for v in obj]
                        if not isinstance(obj, dict):
                            return obj
                        out = {}
                        for key, val in obj.items():
                            if key in ('id', 'item_id', 'call_id') and isinstance(val, str):
                                out[key] = identities.setdefault(val, f'generated_{len(identities)}')
                            elif key in ('created', 'created_at'):
                                out[key] = 0
                            else:
                                out[key] = normalize(val)
                        return out
                    cases[f'{route}/{prompt}/{stream}'] = {'status': result.status,
                                                           'wire': normalize(value)}
        return cases
    finally:
        process.terminate()
        try:
            process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            raise
        if process.returncode:
            raise RuntimeError(f'fixture exit {process.returncode}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', required=True)
    parser.add_argument('--record', action='store_true')
    args = parser.parse_args()
    actual = collect(args.fixture)
    path = Path(__file__).parent / 'golden' / 'api-wire.json'
    if args.record:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(actual, ensure_ascii=False, indent=2) + '\n')
    else:
        expected = json.loads(path.read_text())
        for name, value in actual.items():
            assert value == expected[name], f'wire contract changed: {name}'
        assert actual.keys() == expected.keys()
        print(f'{len(actual)} wire golden cases passed (IDs/timestamps normalized)')
