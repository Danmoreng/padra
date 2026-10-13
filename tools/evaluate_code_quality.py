"""Run frozen code-generation tasks against one local Padra HTTP instance.

Requires Python and Node with TypeScript stripping. Each generated program runs
in a disposable directory with a short timeout and process resource limits.
SQL tasks run on an in-memory fixture with SQLite's read-only query authorizer.
"""
import argparse
import ast
import functools
import hashlib
import json
import os
import pathlib
import re
import resource
import sqlite3
import subprocess
import tempfile
import urllib.request


def limits():
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))


@functools.lru_cache(maxsize=1)
def node_binary():
    # Bypass a package-manager shim which may write caches on each invocation.
    return subprocess.check_output(['node', '-p', 'process.execPath'], text=True).strip()


def check(task, content):
    match = re.search(r'```[^\n]*\n(.*?)```', content, re.S)
    source = match.group(1) if match else content.strip()
    if task['language'] == 'sql':
        db = sqlite3.connect(':memory:')
        db.executescript(task['setup'])
        allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}
        db.set_authorizer(lambda action, *_: sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY)
        return [list(row) for row in db.execute(source).fetchall()] == task['expected']
    if task['language'] == 'python':
        tree = ast.parse(source)
        allowed_imports = {'math', 'collections', 'itertools', 'typing', 're', 'statistics', 'functools'}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(alias.name.split('.')[0] not in allowed_imports for alias in node.names):
                return False
            if isinstance(node, ast.ImportFrom) and (node.module or '').split('.')[0] not in allowed_imports:
                return False
            if isinstance(node, ast.Attribute) and node.attr.startswith('__'):
                return False
            if isinstance(node, ast.Name) and node.id in {'open', 'exec', 'eval', 'compile', '__import__', 'input'}:
                return False
        command, suffix = ['python3', '-B'], '.py'
        program = source + '\n' + task['checks'] + '\n'
    else:
        if re.search(r'\b(import|require|process|fetch|eval)\b', source):
            return False
        command, suffix = [node_binary(), '--experimental-strip-types'], '.mts'
        program = source + '\n' + task['checks'] + '\n'
    with tempfile.TemporaryDirectory(prefix='padra-code-quality-') as directory:
        path = pathlib.Path(directory) / ('solution' + suffix)
        path.write_text(program)
        result = subprocess.run(command + [str(path)], cwd=directory, timeout=5,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                preexec_fn=limits, env={'PATH': os.environ['PATH']})
        return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--tasks', type=pathlib.Path, required=True)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    records = []
    for task in json.loads(args.tasks.read_text()):
        body = {'model': 'mellum21-exl3-k4-research', 'messages': [{'role': 'user', 'content': task['prompt']}],
                'max_completion_tokens': 384, 'temperature': 0, 'enable_thinking': False}
        request = urllib.request.Request(args.base_url.rstrip('/') + '/v1/chat/completions',
                                         json.dumps(body).encode(), {'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=600) as response:
            completion = json.load(response)
        content = completion['choices'][0]['message']['content']
        error = None
        try:
            passed = check(task, content)
        except Exception as exception:
            passed, error = False, str(exception)
        records.append({'name': task['name'], 'language': task['language'], 'passed': passed,
                        'error': error, 'response': completion})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result = {'tasks_sha256': hashlib.sha256(args.tasks.read_bytes()).hexdigest(),
                  'passed': sum(record['passed'] for record in records), 'total': len(records), 'records': records}
        args.output.write_text(json.dumps(result, indent=2))
        print(task['name'], passed, error or '', flush=True)


if __name__ == '__main__':
    main()
