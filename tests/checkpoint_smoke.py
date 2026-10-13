"""Bounded real-checkpoint HTTP checks against an already running Padra server.

No checkpoint conversion, downloads or model loading are performed by this script.
Use --long to cross the local KV ring boundary and verify wrapped snapshot restore.
"""
import argparse
import ast,json,pathlib,time,urllib.request,urllib.error
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--base-url',default='http://127.0.0.1:18080')
parser.add_argument('--output',type=pathlib.Path,default=pathlib.Path('/tmp/padra-checkpoint-smoke'))
parser.add_argument('--long',action='store_true',help='Include >1024-token prefill and wrapped KV restore checks')
args=parser.parse_args()
BASE=args.base_url.rstrip('/');OUT=args.output;OUT.mkdir(parents=True,exist_ok=True)
MODEL='mellum21-exl3-k4-research';records=[]
def request(path,body=None,method=None):
    data=None if body is None else json.dumps(body).encode()
    req=urllib.request.Request(BASE+path,data=data,headers={'Content-Type':'application/json'},method=method)
    return urllib.request.urlopen(req,timeout=600)
def chat(name,messages,**extra):
    body=dict(model=MODEL,messages=messages,max_completion_tokens=64,temperature=0,enable_thinking=False);body.update(extra)
    start=time.monotonic()
    with request('/v1/chat/completions',body) as r:
        result=json.load(r);headers=dict(r.headers)
    entry=dict(name=name,seconds=time.monotonic()-start,request=body,response=result,headers=headers);records.append(entry)
    (OUT/'smoke.json').write_text(json.dumps(records,indent=2))
    print(name,round(entry['seconds'],3),result['usage'],repr(result['choices'][0]['message']['content']),flush=True)
    return result
user=lambda s:[dict(role='user',content=s)]
first=chat('greedy-code',user('Write a Python function add(a, b) that returns their sum. Output only code.'))
code=first['choices'][0]['message']['content'].strip()
if code.startswith('```'):code='\n'.join(code.splitlines()[1:-1])
parsed=ast.parse(code);assert any(isinstance(n,ast.FunctionDef) and n.name=='add' for n in ast.walk(parsed)),code
repeat=chat('greedy-repeat',user('Write a Python function add(a, b) that returns their sum. Output only code.'))
assert first['choices'][0]['message']==repeat['choices'][0]['message']
for label in ['sample-1','sample-2']:
    result=chat(label,user('Name one primary color. Reply with one word only.'),temperature=.7,top_p=.9,top_k=20,seed=123,frequency_penalty=.2,max_completion_tokens=12)
assert records[-1]['response']['choices'][0]['message']==records[-2]['response']['choices'][0]['message']
with request('/v1/sessions',dict(model=MODEL)) as r:session=json.load(r)['id']
history=user('Remember the code name is ORCHID. Reply only OK.')
one=chat('session-first',history,session_id=session,max_completion_tokens=8)
history=history+[one['choices'][0]['message'],dict(role='user',content='What is the code name? Reply with the name only.')]
cached=chat('session-restore',history,session_id=session,max_completion_tokens=12)
assert cached['usage']['prompt_tokens_details']['cached_tokens']>0
with request('/v1/sessions/'+session) as r:info=json.load(r)
assert info['kv_cache_available'];(OUT/'session.json').write_text(json.dumps(info,indent=2))
fresh=chat('session-fresh-control',history,max_completion_tokens=12)
assert cached['choices'][0]['message']==fresh['choices'][0]['message']
with request('/v1/sessions/'+session,method='DELETE') as r:assert json.load(r)['deleted']
msg=user('Count from 1 to 5, separated by commas.')
plain=chat('stream-control',msg,max_completion_tokens=32)
body=dict(model=MODEL,messages=msg,max_completion_tokens=32,temperature=0,enable_thinking=False,stream=True,stream_options={'include_usage':True})
chunks=[];done=False;usage=None;start=time.monotonic()
with request('/v1/chat/completions',body) as r:
    for line in r:
        if not line.startswith(b'data: '):continue
        data=line[6:].strip()
        if data==b'[DONE]':done=True;break
        frame=json.loads(data)
        if frame.get('usage'):usage=frame['usage']
        for choice in frame.get('choices',[]):chunks.append(choice.get('delta',{}).get('content',''))
streamed=''.join(chunks);assert done and usage;assert streamed==plain['choices'][0]['message']['content']
(OUT/'stream.json').write_text(json.dumps(dict(seconds=time.monotonic()-start,content=streamed,usage=usage),indent=2));print('stream',repr(streamed),usage,flush=True)
# Close a live stream and require both cancellation accounting and recovery.
def metric(name):
    with request('/metrics') as r:text=r.read().decode()
    return int(next(line.split()[1] for line in text.splitlines() if line.startswith(name+' ')))
before=metric('mellum_disconnects_total')
cancel_body=dict(model=MODEL,messages=user('Write a complete Python module with 100 numbered functions. Include every function in full.'),max_completion_tokens=1000,temperature=0,enable_thinking=False,stream=True)
with request('/v1/chat/completions',cancel_body) as r:
    for line in r:
        if line.startswith(b'data: ') and b'[DONE]' not in line:
            frame=json.loads(line[6:])
            if any(choice.get('delta',{}).get('content') for choice in frame.get('choices',[])):break
    else:raise AssertionError('disconnect check did not receive a content token')
deadline=time.monotonic()+20
while metric('mellum_disconnects_total')<=before:
    assert time.monotonic()<deadline,'stream disconnect was not accounted for'
    time.sleep(.1)
recovered=chat('disconnect-recovery',user('Count from 1 to 5, separated by commas.'),max_completion_tokens=32)
assert recovered['choices'][0]['message']==plain['choices'][0]['message']
if args.long:
    long=chat('local-ring-wrap',user('# padding line\n'*270+'\nIgnore the comments above and reply with exactly OK.'),max_completion_tokens=8)
    assert long['usage']['prompt_tokens']>1024,long['usage']
    assert long['choices'][0]['message']['content'].strip()=='OK',long
    with request('/v1/sessions',dict(model=MODEL)) as r:session=json.load(r)['id']
    history=user('# padding line\n'*270+'\nRemember that the code name is ORCHID. Reply only OK.')
    first=chat('wrapped-session-first',history,session_id=session,max_completion_tokens=8)
    history += [first['choices'][0]['message'],dict(role='user',content='What is the code name? Reply with the name only.')]
    cached=chat('wrapped-session-restore',history,session_id=session,max_completion_tokens=8)
    assert cached['usage']['prompt_tokens_details']['cached_tokens']>1024
    fresh=chat('wrapped-session-fresh',history,max_completion_tokens=8)
    assert cached['choices'][0]['message']==fresh['choices'][0]['message']
    assert fresh['choices'][0]['message']['content'].strip()=='ORCHID'
    with request('/v1/sessions/'+session,method='DELETE') as r:assert json.load(r)['deleted']
with request('/metrics') as r:metrics=r.read().decode()
(OUT/'metrics_after_smoke.txt').write_text(metrics)
assert 'mellum_model_loads_total 1\n' in metrics and 'mellum_weight_uploads_total 1\n' in metrics
with request('/v1/models') as r:models=json.load(r)
if '-simt-' in models['data'][0]['execution_profile']:
    profile=models['data'][0]['execution_profile']
    captures=5 if '-attention-tiered-' in profile else 3 if '-layer-batch256-' in profile else 2
    assert f'mellum_graph_captures_total {captures}\n' in metrics
with request('/readyz') as r:assert json.load(r)['ready']
print('ALL SMOKE CHECKS PASSED',flush=True)
