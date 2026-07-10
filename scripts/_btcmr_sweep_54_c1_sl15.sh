#!/bin/bash
# Sweep 54 BTC-MR configs: full 365d + fold3-of-4. CONFIRM=1 SL_ATR=1.5 fixed.
cd /Users/ashish/Documents/beast-trader || exit 1
OUT=/tmp/btcmr_sweep54_c1_sl15.jsonl
: > "$OUT"
for BB in 2 2.5 3; do
  for RL in 10 20 30; do
    case $RL in 10) RH=90;; 20) RH=80;; 30) RH=70;; esac
    for ADX in 20 25 30; do
      for TS in 16 24; do
        ENV="BTCMR_CONFIRM=1 BTCMR_SL_ATR=1.5 BTCMR_BB_MULT=$BB BTCMR_RSI_LOW=$RL BTCMR_RSI_HIGH=$RH BTCMR_ADX_MAX=$ADX BTCMR_TIME_STOP_BARS=$TS"
        FULL=$(env $ENV python3 -B scripts/_btcmr_run.py --symbol BTCUSD --days 365 2>/dev/null | tail -1)
        F3=$(env $ENV python3 -B scripts/_btcmr_run.py --symbol BTCUSD --days 365 --folds 4 --fold 3 2>/dev/null | tail -1)
        echo "{\"bb\":$BB,\"rl\":$RL,\"rh\":$RH,\"adx\":$ADX,\"ts\":$TS,\"full\":$FULL,\"fold3\":$F3}" >> "$OUT"
        echo "done bb=$BB rl=$RL adx=$ADX ts=$TS"
      done
    done
  done
done
echo "SWEEP COMPLETE"
