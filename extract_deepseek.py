import json, glob, sys

files = glob.glob('runs/tournament/*/ep*.json') + glob.glob('runs/match_tournament/*/ep*.json')
MODEL = 'deepseek-v4-pro'

decisions = []
for f in files:
    try:
        d = json.load(open(f))
    except Exception:
        continue
    for s in d.get('steps', []):
        if s.get('agent_name') != MODEL:
            continue
        pub = s['observation'].get('public', {})
        priv = s['observation'].get('private', {})
        raw = s.get('response', {})
        raw = raw.get('raw_output') if isinstance(raw, dict) else None
        decisions.append({
            'file': f,
            'step': s.get('step'),
            'hole': priv.get('hole'),
            'board': pub.get('board'),
            'street': pub.get('street'),
            'pot': pub.get('pot'),
            'to_call': pub.get('to_call'),
            'your_stack': pub.get('your_stack'),
            'opp_stack': pub.get('opp_stack'),
            'street_commit': pub.get('your_street_commit'),
            'position': pub.get('position'),
            'action': s.get('selected_action'),
            'amount': s.get('selected_amount'),
            'invalid': s.get('invalid'),
            'raw': raw or '',
        })

# prioritize: facing a bet, then larger pot
facing = [x for x in decisions if (x['to_call'] or 0) > 0]
facing.sort(key=lambda x: -(x['pot'] or 0))
notfacing = [x for x in decisions if (x['to_call'] or 0) == 0]
notfacing.sort(key=lambda x: -(x['pot'] or 0))

selected = facing[:32] + notfacing[:8]

print(f"TOTAL deepseek decisions: {len(decisions)}; facing-bet: {len(facing)}; selected: {len(selected)}")
print("="*100)
for i, x in enumerate(selected):
    print(f"\n########## DECISION {i+1}  [{x['file']} step{x['step']}] ##########")
    print(f"hole={x['hole']} board={x['board']} street={x['street']} pos={x['position']}")
    print(f"pot={x['pot']} to_call={x['to_call']} your_stack={x['your_stack']} opp_stack={x['opp_stack']} street_commit={x['street_commit']}")
    print(f">>> ACTION: {x['action']} amount={x['amount']} invalid={x['invalid']}")
    print("--- REASONING ---")
    print(x['raw'][:2600])
    print("-"*80)
