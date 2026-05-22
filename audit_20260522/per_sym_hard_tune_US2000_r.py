"""2026-05-23 — Hard-tune US2000.r per-symbol with mirror-aware BT.

Mirror-aware BT (backtest/v5_backtest.py) reads live config for:
  - POST_BIG_WIN_COOLDOWN_*
  - LOSS_STREAK (via params)
  - PULLBACK_ATR_RETRACE_PER_SYMBOL / _MAX_WAIT_BARS_PER_SYMBOL
  - VWAP_BUFFER_PER_SYMBOL
  - TRAIL_OVERRIDE_REGIME (v5_backtest module-level)
  - DIR_BIAS / _DIR_BIAS_REGIME_STR (v5_backtest module-level)

Strategy:
  Account equity $1,219 @ risk_pct=2.0, start_equity=1219.
  9 dims sweep but coordinate-descent to keep BT count low.

  Phase A — coordinate-descent independent sweeps (each dim, others at live):
    SL: 9 vals, Trail: 7 profiles, mq: 8 vals, pb_atr: 5, pb_wait: 3,
    vwap: 6, pbw_cd: 5, ls_cd: 5, dir_bias_per_regime: 9 combos
    Total Phase A ≈ 57 BTs

  Phase B — top-2 per dim → cartesian combine, capped 200 BTs

  Phase C — WF 5-fold on top-3 from Phase B

Ship rule:  Δ ≥ $30 AND ≥3/5 WF folds positive AND avg PF ≥ 1.5
"""
import sys
import os
import json
import time
import itertools
from multiprocessing import Pool, cpu_count

sys.path.insert(0, '/Users/ashish/Documents/beast-trader')

SYMBOL = 'US2000.r'
OUT_DIR = '/Users/ashish/Documents/beast-trader/audit_20260522'
RISK_PCT = 2.0
START_EQUITY = 1219.0

# ───── Worker init ─────
# Each worker patches module-level state before running BT so we can sweep
# config-driven knobs (pullback, vwap, post-big-win, trail, dir_bias).

def _apply_overrides(symbol, sl, trail_steps, mq_val, pb_atr, pb_wait,
                     vwap_buf, pbw_secs, ls_secs, dir_bias_regime):
    """Apply in-memory overrides to config + v5_backtest module-level dicts.
    Returns a params dict that backtest_symbol() will receive.
    """
    import config as _cfg
    import backtest.v5_backtest as _bt

    # Per-symbol PULLBACK overrides — patch the PER_SYMBOL dict
    _cfg.PULLBACK_ATR_RETRACE_PER_SYMBOL = dict(getattr(_cfg, 'PULLBACK_ATR_RETRACE_PER_SYMBOL', {}))
    _cfg.PULLBACK_ATR_RETRACE_PER_SYMBOL[symbol] = float(pb_atr)
    _cfg.PULLBACK_MAX_WAIT_BARS_PER_SYMBOL = dict(getattr(_cfg, 'PULLBACK_MAX_WAIT_BARS_PER_SYMBOL', {}))
    _cfg.PULLBACK_MAX_WAIT_BARS_PER_SYMBOL[symbol] = int(pb_wait)

    # VWAP buffer override
    _cfg.VWAP_BUFFER_PER_SYMBOL = dict(getattr(_cfg, 'VWAP_BUFFER_PER_SYMBOL', {}))
    _cfg.VWAP_BUFFER_PER_SYMBOL[symbol] = float(vwap_buf)

    # POST_BIG_WIN cooldown seconds
    _cfg.POST_BIG_WIN_COOLDOWN_SECS = int(pbw_secs)
    _cfg.POST_BIG_WIN_COOLDOWN_ENABLED = True
    # LOSS_STREAK cooldown seconds (translated to bars below)
    _cfg.LOSS_STREAK_COOLDOWN_SECS = int(ls_secs)

    # Patch v5_backtest module-level overrides
    # 1. Trail regime override — clear the regime-level dict for this symbol so
    #    `force_trail` (via params) takes effect. BT does
    #    TRAIL_OVERRIDE_REGIME.get(symbol, {}).get(regime) and expects dict.
    if trail_steps is not None:
        _bt.TRAIL_OVERRIDE_REGIME = dict(getattr(_bt, 'TRAIL_OVERRIDE_REGIME', {}))
        _bt.TRAIL_OVERRIDE_REGIME[symbol] = {}  # empty dict → no regime override
        # Also clear TRAIL_OVERRIDE for symbol so `force_trail` is used
        _bt.TRAIL_OVERRIDE = dict(getattr(_bt, 'TRAIL_OVERRIDE', {}))
        _bt.TRAIL_OVERRIDE.pop(symbol, None)

    # 2. SL override — clear regime-level so params.sl_atr_mult is used
    _bt.SL_OVERRIDE = dict(getattr(_bt, 'SL_OVERRIDE', {}))
    _bt.SL_OVERRIDE.pop(symbol, None)  # → falls back to p["sl_atr_mult"]
    _bt.SL_OVERRIDE_REGIME = dict(getattr(_bt, 'SL_OVERRIDE_REGIME', {}))
    _bt.SL_OVERRIDE_REGIME.pop(symbol, None)

    # 3. Direction bias per-regime override.
    # dir_bias_regime is a dict like {'trending':'LONG','ranging':'BOTH',...}
    # Patch module-level _DIR_BIAS_REGIME_STR and DIR_BIAS.
    _bt._DIR_BIAS_REGIME_STR = dict(getattr(_bt, '_DIR_BIAS_REGIME_STR', {}))
    if dir_bias_regime:
        _bt._DIR_BIAS_REGIME_STR[symbol] = dict(dir_bias_regime)
    else:
        _bt._DIR_BIAS_REGIME_STR.pop(symbol, None)
    # Also strip per-symbol DIR_BIAS so regime dict is the only source.
    _bt.DIR_BIAS = dict(getattr(_bt, 'DIR_BIAS', {}))
    _bt.DIR_BIAS.pop(symbol, None)

    # 4. SIGNAL_QUALITY_SYMBOL — patch per-symbol min_quality (all regimes).
    _cfg.SIGNAL_QUALITY_SYMBOL = dict(getattr(_cfg, 'SIGNAL_QUALITY_SYMBOL', {}))
    _cfg.SIGNAL_QUALITY_SYMBOL[symbol] = {
        'trending': int(mq_val), 'ranging': int(mq_val),
        'volatile': int(mq_val), 'low_vol': int(mq_val),
    }

    # Build params dict
    params = {
        'sl_atr_mult': float(sl),
        'risk_pct': RISK_PCT,
        'start_equity': START_EQUITY,
        'min_quality': {'trending': int(mq_val), 'ranging': int(mq_val),
                        'volatile': int(mq_val), 'low_vol': int(mq_val)},
        'loss_streak_enabled': True,
        'loss_streak_count': 2,
        'loss_streak_window_bars': 4,
        'loss_streak_cooldown_bars': max(1, int(round(ls_secs / 3600.0))),
    }
    if trail_steps is not None:
        params['force_trail'] = trail_steps
    return params


def _run_one(args):
    """Worker for one BT. args is (tag, days, override_kwargs)."""
    tag, days, kw = args
    try:
        params = _apply_overrides(SYMBOL, **kw)
        from backtest.v5_backtest import backtest_symbol
        r = backtest_symbol(SYMBOL, days=days, params=params, verbose=False)
        if r is None:
            return tag, None
        return tag, {
            'trades': r['trades'],
            'pf': float(r['pf']) if r['pf'] != 999 else 999.0,
            'pnl': float(r['pnl']),
            'wr': float(r.get('wr', 0)),
            'dd': float(r.get('dd', 0)),
            'avg_r': float(r.get('avg_r', 0)),
        }
    except Exception as e:
        return tag, {'error': str(e)[:120]}


# ───── Live config baseline ─────
LIVE_BASE = {
    'sl': 0.2, 'trail': '_TIGHT_LOCK', 'mq': 25,
    'pb_atr': 0.9, 'pb_wait': 5, 'vwap_buf': 1.5,
    'pbw_secs': 3600, 'ls_secs': 18000,
    'dir_bias': {'ranging': 'LONG', 'volatile': 'LONG'},
}


def _bias_dict_for_tag(tag):
    """Direction bias presets for sweep:
       'live'       — LIVE_BASE['dir_bias']
       'BOTH_ALL'   — no bias
       'LONG_ALL'   — LONG in every regime
       'LONG_TR'    — LONG only in trending
       'LONG_RA'    — LONG only in ranging
       'LONG_VO'    — LONG only in volatile
       'LONG_TR_RA' — LONG in trending+ranging
       'LONG_TR_VO' — LONG in trending+volatile
       'LONG_RA_VO' — LONG in ranging+volatile (== LIVE_BASE)
    """
    if tag == 'live':
        return dict(LIVE_BASE['dir_bias'])
    if tag == 'BOTH_ALL':
        return {}
    if tag == 'LONG_ALL':
        return {r: 'LONG' for r in ('trending','ranging','volatile','low_vol')}
    if tag == 'LONG_TR':
        return {'trending': 'LONG'}
    if tag == 'LONG_RA':
        return {'ranging': 'LONG'}
    if tag == 'LONG_VO':
        return {'volatile': 'LONG'}
    if tag == 'LONG_TR_RA':
        return {'trending': 'LONG', 'ranging': 'LONG'}
    if tag == 'LONG_TR_VO':
        return {'trending': 'LONG', 'volatile': 'LONG'}
    if tag == 'LONG_RA_VO':
        return {'ranging': 'LONG', 'volatile': 'LONG'}
    return {}


def _trail_obj(name):
    """Resolve trail name to backtest tuple list."""
    import auto_tuned as _at
    return {
        '_TIGHT_LOCK':       _at._TIGHT_LOCK,
        '_AGGR_LOCK':        _at._AGGR_LOCK,
        '_RUNNER_NO_BE':     _at._RUNNER_NO_BE,
        '_WIDE_RUNNER':      _at._WIDE_RUNNER,
        '_RANGE_TIGHT':      _at._RANGE_TIGHT,
        '_TREND_LOOSE':      _at._TREND_LOOSE,
        '_WIDE_RUNNER_BE07': _at._WIDE_RUNNER_BE07,
    }[name]


def _base_kw(overrides):
    """Build kw dict from LIVE_BASE + overrides."""
    o = dict(LIVE_BASE)
    o.update(overrides)
    db = o['dir_bias']
    if isinstance(db, str):
        db = _bias_dict_for_tag(db)
    return {
        'sl': o['sl'],
        'trail_steps': _trail_obj(o['trail']),
        'mq_val': o['mq'],
        'pb_atr': o['pb_atr'],
        'pb_wait': o['pb_wait'],
        'vwap_buf': o['vwap_buf'],
        'pbw_secs': o['pbw_secs'],
        'ls_secs': o['ls_secs'],
        'dir_bias_regime': dict(db),
    }


# ───── PHASE A: per-dim independent sweeps ─────
DIMS = {
    'sl':       [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0],
    'trail':    ['_TIGHT_LOCK', '_AGGR_LOCK', '_RUNNER_NO_BE', '_WIDE_RUNNER',
                 '_RANGE_TIGHT', '_TREND_LOOSE', '_WIDE_RUNNER_BE07'],
    'mq':       [22, 25, 28, 30, 33, 35, 38, 40],
    'pb_atr':   [0.4, 0.6, 0.8, 1.0, 1.2],
    'pb_wait':  [3, 5, 7],
    'vwap_buf': [0.0, 0.3, 0.5, 0.7, 1.0, 1.5],
    'pbw_secs': [1800, 3600, 5400, 7200, 10800],
    'ls_secs':  [3600, 7200, 10800, 14400, 18000],
    'dir_bias': ['live', 'BOTH_ALL', 'LONG_ALL', 'LONG_TR', 'LONG_RA',
                 'LONG_VO', 'LONG_TR_RA', 'LONG_TR_VO', 'LONG_RA_VO'],
}


def phase_a():
    """Independent sweep each dim. Returns dict[dim] = [(tag, result), ...] sorted by pnl desc."""
    tasks = []
    for dim, vals in DIMS.items():
        for v in vals:
            kw = _base_kw({dim: v})
            tag = f'A:{dim}={v!r}'
            tasks.append((tag, 180, kw))

    # Baseline (live config)
    tasks.append(('A:BASELINE', 180, _base_kw({})))

    with Pool(min(8, max(2, cpu_count() - 2))) as pool:
        results = dict(pool.map(_run_one, tasks))

    base = results.get('A:BASELINE') or {}

    # Group by dim
    by_dim = {}
    for dim, vals in DIMS.items():
        rows = []
        for v in vals:
            tag = f'A:{dim}={v!r}'
            r = results.get(tag)
            if r and not r.get('error'):
                rows.append({'value': v, **r})
        rows.sort(key=lambda r: -r['pnl'])
        by_dim[dim] = rows
    return base, by_dim


# ───── PHASE B: combine top-2 per dim (cartesian, capped 200) ─────
def phase_b(by_dim, base_pnl, cap=200):
    """Top-2 per dim → cartesian → capped 200 BTs."""
    top = {dim: [r['value'] for r in rows[:2]] for dim, rows in by_dim.items()}
    # Cartesian product
    combos = list(itertools.product(
        top['sl'], top['trail'], top['mq'], top['pb_atr'], top['pb_wait'],
        top['vwap_buf'], top['pbw_secs'], top['ls_secs'], top['dir_bias'],
    ))
    # 2^9 = 512 max → trim by greedy. Take only 200 lowest "distance from
    # best single-dim winner" by ranking on sum of indices.
    if len(combos) > cap:
        # Order each top list so index 0 = winner. Score combo by sum of indices.
        def score(c):
            return sum(
                top[dim].index(val)
                for dim, val in zip(
                    ('sl','trail','mq','pb_atr','pb_wait','vwap_buf','pbw_secs','ls_secs','dir_bias'),
                    c,
                )
            )
        combos.sort(key=score)
        combos = combos[:cap]

    tasks = []
    for i, c in enumerate(combos):
        sl, trail, mq, pa, pw, vw, pbw, ls, db = c
        kw = _base_kw({
            'sl': sl, 'trail': trail, 'mq': mq, 'pb_atr': pa, 'pb_wait': pw,
            'vwap_buf': vw, 'pbw_secs': pbw, 'ls_secs': ls, 'dir_bias': db,
        })
        tasks.append((f'B{i}', 180, kw))

    with Pool(min(8, max(2, cpu_count() - 2))) as pool:
        results = dict(pool.map(_run_one, tasks))

    rows = []
    for i, c in enumerate(combos):
        r = results.get(f'B{i}')
        if r and not r.get('error'):
            sl, trail, mq, pa, pw, vw, pbw, ls, db = c
            rows.append({
                'tag': f'B{i}',
                'config': {
                    'sl': sl, 'trail': trail, 'mq': mq, 'pb_atr': pa,
                    'pb_wait': pw, 'vwap_buf': vw, 'pbw_secs': pbw,
                    'ls_secs': ls, 'dir_bias': db,
                },
                **r,
                'delta': r['pnl'] - base_pnl,
            })
    rows.sort(key=lambda x: -x['pnl'])
    return rows


# ───── PHASE C: WF 5-fold on top-3 ─────
def phase_c(top_b, base_pnl):
    """5-fold WF (60,90,120,150,180) on top-3 from Phase B."""
    if not top_b:
        return []
    folds = [60, 90, 120, 150, 180]
    tasks = []
    for j, row in enumerate(top_b[:3]):
        c = row['config']
        kw = _base_kw({
            'sl': c['sl'], 'trail': c['trail'], 'mq': c['mq'],
            'pb_atr': c['pb_atr'], 'pb_wait': c['pb_wait'],
            'vwap_buf': c['vwap_buf'], 'pbw_secs': c['pbw_secs'],
            'ls_secs': c['ls_secs'], 'dir_bias': c['dir_bias'],
        })
        # Baseline fold tasks (live config)
        base_kw = _base_kw({})
        for d in folds:
            tasks.append((f'C{j}|cand|f{d}', d, kw))
            tasks.append((f'C{j}|base|f{d}', d, base_kw))

    with Pool(min(8, max(2, cpu_count() - 2))) as pool:
        results = dict(pool.map(_run_one, tasks))

    summary = []
    for j, row in enumerate(top_b[:3]):
        c = row['config']
        fold_rows = []
        pos = 0
        pf_sum = 0.0
        pf_n = 0
        for d in folds:
            cand = results.get(f'C{j}|cand|f{d}') or {}
            base = results.get(f'C{j}|base|f{d}') or {}
            cand_pnl = cand.get('pnl', 0) or 0
            base_pnl_f = base.get('pnl', 0) or 0
            delta = cand_pnl - base_pnl_f
            fold_rows.append({
                'days': d,
                'cand_pnl': round(cand_pnl, 2),
                'base_pnl': round(base_pnl_f, 2),
                'delta': round(delta, 2),
                'pf': round(float(cand.get('pf', 0) or 0), 2),
                'wr': round(float(cand.get('wr', 0) or 0), 2),
                'n': int(cand.get('trades', 0) or 0),
            })
            if delta > 0:
                pos += 1
            pf_val = float(cand.get('pf', 0) or 0)
            if pf_val > 0 and pf_val != 999:
                pf_sum += pf_val
                pf_n += 1
        avg_pf = pf_sum / max(1, pf_n) if pf_n else 0.0
        delta_180 = row['delta']
        ship = (delta_180 >= 30) and (pos >= 3) and (avg_pf >= 1.5)
        summary.append({
            'rank': j + 1,
            'tag': row['tag'],
            'config': c,
            'full_180d_pnl': round(row['pnl'], 2),
            'full_180d_delta': round(delta_180, 2),
            'wf_folds': fold_rows,
            'positive_folds': pos,
            'avg_pf': round(avg_pf, 2),
            'ship': bool(ship),
        })
    return summary


# ───── MAIN ─────
def main():
    t0 = time.time()
    print(f'=== Hard-tune {SYMBOL} mirror-aware BT @ risk_pct={RISK_PCT}, equity=${START_EQUITY} ===')

    # Phase A
    print('Phase A — independent dim sweeps...')
    pa_t = time.time()
    base, by_dim = phase_a()
    pa_count = sum(len(v) for v in DIMS.values()) + 1
    print(f'  Phase A: {pa_count} BTs in {time.time()-pa_t:.1f}s')
    if not base:
        print('  Baseline failed!')
        return
    base_pnl = base['pnl']
    print(f'  Baseline: PF={base["pf"]:.2f} PnL=${base["pnl"]:.2f} N={base["trades"]} WR={base["wr"]:.1f}%')
    for dim, rows in by_dim.items():
        top3 = rows[:3]
        line = f'  {dim:9s} top3: ' + ' | '.join(
            f'{r["value"]!r:>12}: ${r["pnl"]:>10.2f} PF{r["pf"]:>4.1f}'
            for r in top3
        )
        print(line)

    # Phase B
    print('\nPhase B — top-2 combine (cap 200)...')
    pb_t = time.time()
    b_results = phase_b(by_dim, base_pnl, cap=200)
    print(f'  Phase B: {len(b_results)} valid BTs in {time.time()-pb_t:.1f}s')
    if not b_results:
        print('  No phase B results!')
        return
    for r in b_results[:5]:
        c = r['config']
        print(f"  {r['tag']:5s}: ${r['pnl']:>10.2f} PF{r['pf']:>4.1f} Δ${r['delta']:>+10.2f}  sl={c['sl']} trl={c['trail']:15s} mq={c['mq']} pa={c['pa_atr'] if False else c['pb_atr']} pw={c['pb_wait']} vw={c['vwap_buf']} pbw={c['pbw_secs']} ls={c['ls_secs']} db={c['dir_bias']}")

    # Phase C
    print('\nPhase C — WF 5-fold on top-3...')
    pc_t = time.time()
    c_summary = phase_c(b_results, base_pnl)
    print(f'  Phase C: {len(c_summary)*10} BTs in {time.time()-pc_t:.1f}s')

    winner = None
    for s in c_summary:
        ship_str = 'SHIP' if s['ship'] else 'HOLD'
        print(f"  rank{s['rank']} {ship_str}: Δ${s['full_180d_delta']:>+10.2f} pos={s['positive_folds']}/5 avgPF={s['avg_pf']:.2f}")
        for f in s['wf_folds']:
            print(f"    d{f['days']:3d}: cand=${f['cand_pnl']:>+10.2f} base=${f['base_pnl']:>+10.2f} Δ=${f['delta']:>+10.2f} PF={f['pf']:.2f} n={f['n']}")
        if s['ship'] and winner is None:
            winner = s

    elapsed = time.time() - t0
    out = {
        'symbol': SYMBOL,
        'risk_pct': RISK_PCT,
        'start_equity': START_EQUITY,
        'live_baseline_config': LIVE_BASE,
        'baseline': base,
        'phase_a_top3_per_dim': {dim: rows[:3] for dim, rows in by_dim.items()},
        'phase_b_top5': b_results[:5],
        'phase_c_top3': c_summary,
        'winner': winner,
        'recommendation': 'SHIP' if winner else 'HOLD',
        'total_runtime_s': round(elapsed, 1),
        'total_backtests': pa_count + len(b_results) + len(c_summary) * 10,
    }
    out_path = os.path.join(OUT_DIR, f'per_sym_hard_tune_{SYMBOL}.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f'\n=== DONE in {elapsed:.1f}s — wrote {out_path} ===')
    print(f"Recommendation: {out['recommendation']}")
    if winner:
        c = winner['config']
        print(f"Winner: sl={c['sl']} trl={c['trail']} mq={c['mq']} pa={c['pb_atr']} pw={c['pb_wait']} vw={c['vwap_buf']} pbw={c['pbw_secs']} ls={c['ls_secs']} db={c['dir_bias']}")
        print(f"  Δ ${winner['full_180d_delta']:.2f}  positive_folds={winner['positive_folds']}/5  avg_pf={winner['avg_pf']:.2f}")
    return out


if __name__ == '__main__':
    main()
