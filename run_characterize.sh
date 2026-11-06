#!/usr/bin/env bash
# Characterize a lantern preset at its 41 native wavelengths, within a memory budget.
#
#   CBEAM_LANTERN=pl19_1200 bash run_characterize.sh
#
# Runs the first wavelength alone to measure its peak memory, then the rest
# JOBS at a time, each single-threaded. JOBS is the number of characterizations
# whose peak (plus 30%) fits in MEM_BUDGET_GB (default: half of this host's
# RAM), capped at the cores minus 2. Logs go to logs/<lantern>/.
set -euo pipefail
cd "$(dirname "$0")"
: "${CBEAM_LANTERN:?set CBEAM_LANTERN to a batch_pipeline.LANTERNS name}"
export CBEAM_LANTERN
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CBEAM_BACKEND=numpy
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
LOGS=logs/$CBEAM_LANTERN
mkdir -p "$LOGS"

total_gb=$(awk '/MemTotal/ {printf "%d", $2 / 2^20}' /proc/meminfo)
MEM_BUDGET_GB=${MEM_BUDGET_GB:-$(( total_gb / 2 ))}

# 1. one wavelength alone; characterize.py reports its peak resident memory
python characterize.py --lantern "$CBEAM_LANTERN" 20 > "$LOGS/char_20.log" 2>&1
peak=$(awk '/PEAK_GB/ {print $2}' "$LOGS/char_20.log")
JOBS=${JOBS:-$(python -c "import math,os; print(max(1, min(os.cpu_count()-2, math.floor($MEM_BUDGET_GB / ($peak * 1.3)))))")}
echo "lantern $CBEAM_LANTERN: characterization peak ${peak} GB, budget ${MEM_BUDGET_GB} GB -> $JOBS parallel" | tee "$LOGS/plan.txt"

# 2. the other 40 (already-cached ones are skipped by characterize.py)
seq 0 40 | grep -vx 20 | xargs -P "$JOBS" -I{} sh -c \
    'python characterize.py --lantern "$CBEAM_LANTERN" {} > "'"$LOGS"'/char_{}.log" 2>&1 || echo {}'     > "$LOGS/failed.txt"
# 3. retry the failed ones once (e.g. a transient Julia start-up error)
while read -r i; do
    python characterize.py --lantern "$CBEAM_LANTERN" "$i" >> "$LOGS/rerun.log" 2>&1 || echo "FAILED AGAIN $i" | tee -a "$LOGS/plan.txt"
done < "$LOGS/failed.txt"
echo "ALLDONE $CBEAM_LANTERN" | tee -a "$LOGS/plan.txt"
