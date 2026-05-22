import json
d = json.load(open('/Users/ashish/Documents/beast-trader/audit_20260522/avg_win_cap_sim.json'))
syms = sorted(d['per_symbol'].items(), key=lambda x: -x[1]['tot_loss'])
print(f'{"SYM":11} {"nL":>3} {"totL":>7} {"avgL":>6} {"maxL":>6} {"avgW":>6} {"nWavg":>5} | 2.0/2.0 fires save | 1.0/1.0 fires save | 0.75/0.5 fires save')
for sym, s in syms[:30]:
    print(f'{sym:11} {s["n_losses"]:3d} {s["tot_loss"]:7.2f} {s["avg_loss"]:6.2f} {s["max_loss"]:6.2f} {s["avg_avg_win"]:6.2f} {s["n_with_avg_win"]:5d} | {s["fires_at_2.0_2.0"]:5d} {s["saved_at_2.0_2.0"]:6.2f} | {s["fires_at_1.0_1.0"]:5d} {s["saved_at_1.0_1.0"]:6.2f} | {s["fires_at_0.75_0.5"]:5d} {s["saved_at_0.75_0.5"]:6.2f}')
print()
print('Category summary:')
for c, v in sorted(d['per_category_summary'].items(), key=lambda x: -x[1]['loss']):
    print(f'  {c:18} n={v["n"]:4d}  loss=${v["loss"]:7.2f}')

# Now break down sweep by category
print()
print('Category-stratified savings (which layer would AvgWinLossCap pre-empt?):')
print(f'{"MULT":>5} {"floor":>6} {"TrailSL":>15} {"EarlyLossCut":>15} {"Guardian":>15} {"DailyKill":>14} {"DragonRev":>14} {"EmergDD":>10} {"PeakGB":>10}')
for sw in d['sweep']:
    if (sw['MULT'], sw['floor']) not in [(2.0, 2.0), (1.5, 1.0), (1.25, 1.0), (1.0, 1.0), (1.0, 0.5), (0.75, 0.5)]:
        continue
    bc = sw['by_category']
    def fmt(c):
        v = bc.get(c, {'fires': 0, 'saved': 0})
        return f"{v['fires']:3d} ${v['saved']:6.2f}"
    print(f"{sw['MULT']:5.2f} {sw['floor']:6.2f}  {fmt('TrailSL'):>15}  {fmt('EarlyLossCut'):>15}  {fmt('Guardian'):>15}  {fmt('DailyKillSwitch'):>14}  {fmt('DragonReversal'):>14}  {fmt('EmergencyDD'):>10}  {fmt('PeakGiveback'):>10}")

# Sweep summary subset
print()
print('Sweep summary (focus rows):')
for s in d['sweep']:
    if (s['MULT'], s['floor']) in [(2.0, 2.0), (1.5, 1.0), (1.25, 1.0), (1.0, 1.0), (1.0, 0.5), (0.75, 0.5), (0.5, 0.5)]:
        print(f"  MULT={s['MULT']:.2f} floor=${s['floor']:.2f}  fires={s['fires']:3d}/{s['total_losses']}  saved=${s['saved']:7.2f}  ({s['save_pct']:.1f}% of ${s['total_loss']:.2f})")
