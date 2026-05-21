"""Build actionable-only summary from regime_dir_risk.json."""
import json
d = json.load(open('/Users/ashish/Documents/beast-trader/tune_session_20260521/regime_dir_risk.json'))

dir_changes = []
risk_changes = []

BASE_RISK = 0.8
for sym, regs in sorted(d.items()):
    for regime, cell in sorted(regs.items()):
        db = cell.get('dir_bias', {})
        rc = cell.get('risk_cap', {})
        if db.get('recommend') and db.get('wf_passed') and db['recommend'] != db.get('current'):
            dir_changes.append({
                "sym": sym, "regime": regime,
                "current": db.get('current'), "recommend": db['recommend'],
                "long_pnl": db['long_pnl'], "short_pnl": db['short_pnl'],
                "long_pf": db['long_pf'], "short_pf": db['short_pf'],
                "long_n": db['long_n'], "short_n": db['short_n'],
            })
        if rc.get('recommend') is not None and rc.get('wf_passed'):
            cur = rc.get('current')
            rec = rc['recommend']
            if cur is None and rec >= BASE_RISK:
                continue
            if rec == cur:
                continue
            cs = rc.get('current_score') or 0
            ns = rc.get('score') or 0
            if cs > 0 and ns < cs * 1.10:
                continue
            risk_changes.append({
                "sym": sym, "regime": regime,
                "current": cur, "recommend": rec,
                "score_old": cs, "score_new": ns,
                "pnl": rc['pnl'], "pf": rc['pf'], "trades": rc['trades'],
            })

summary = {
    "summary": {
        "symbols_tested": len(d),
        "regimes": ["trending","ranging","volatile","low_vol"],
        "days": 180,
        "wf_folds": 5,
        "dir_changes": len(dir_changes),
        "risk_changes": len(risk_changes),
    },
    "dir_changes": dir_changes,
    "risk_changes": risk_changes,
}
open('/Users/ashish/Documents/beast-trader/tune_session_20260521/regime_dir_risk_actionable.json','w').write(json.dumps(summary, indent=2))
print(f"dir_changes: {len(dir_changes)}")
print(f"risk_changes: {len(risk_changes)}")
