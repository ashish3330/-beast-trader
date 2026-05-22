import json
d = json.load(open('/Users/ashish/Documents/beast-trader/audit_20260522/avg_win_cap_sim.json'))
print(f'Total losses (30d): {d["n_losses_total"]}, with avg_win data: {d["n_with_avg_win"]}')
print()
print('Distribution of |pnl| at each (MULT, floor):')
for sw in d['sweep']:
    if (sw['MULT'], sw['floor']) not in [(2.0, 2.0), (1.5, 1.0), (1.25, 1.0), (1.0, 1.0), (1.0, 0.5), (0.75, 0.5)]:
        continue
    fired_pnl = []
    skipped_pnl = []
    fired_caps = []
    for r in d['rows']:
        if r['avg_win_at_close'] <= 0:
            continue
        cap = max(r['avg_win_at_close'] * sw['MULT'], sw['floor'])
        if abs(r['pnl']) >= cap:
            fired_pnl.append(abs(r['pnl']))
            fired_caps.append(cap)
        else:
            skipped_pnl.append(abs(r['pnl']))
    fp = sorted(fired_pnl)
    fc = sorted(fired_caps)
    sp = sorted(skipped_pnl)
    if fp:
        print(f"  MULT={sw['MULT']:.2f} floor=${sw['floor']:.2f}  fires={len(fp):3d}/{len(fp)+len(sp)} (with-avg-win-data)")
        print(f"    fired_loss   median=${fp[len(fp)//2]:.2f}  p10=${fp[len(fp)//10]:.2f}  p90=${fp[len(fp)*9//10]:.2f}  max=${fp[-1]:.2f}")
        print(f"    cap          median=${fc[len(fc)//2]:.2f}  p10=${fc[len(fc)//10]:.2f}  p90=${fc[len(fc)*9//10]:.2f}")
        if sp:
            print(f"    skipped_loss median=${sp[len(sp)//2]:.2f}  p90=${sp[len(sp)*9//10]:.2f}  max=${sp[-1]:.2f}")
        print()

# Look at how often the cap fires per symbol with relation to avg loss multiple
print()
print('Per-symbol L/W ratio and savings projection (MULT=1.0, floor=$1.0):')
syms = sorted(d['per_symbol'].items(), key=lambda x: -x[1]['tot_loss'])
for sym, s in syms[:20]:
    if s['n_with_avg_win'] == 0:
        continue
    ratio = s['avg_loss'] / max(s['avg_avg_win'], 0.001)
    n_fires = s['fires_at_1.0_1.0']
    pct_caught = 100.0 * n_fires / s['n_losses'] if s['n_losses'] else 0
    print(f"  {sym:11} L/W={ratio:5.2f}x  fires={n_fires:3d}/{s['n_losses']} ({pct_caught:.0f}%)  saved=${s['saved_at_1.0_1.0']:6.2f}/{s['tot_loss']:.2f}")
