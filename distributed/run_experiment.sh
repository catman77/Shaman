#!/bin/bash
# =============================================================================
# Distributed Shaman Experiment - Local Test Runner
# =============================================================================
#
# Этот скрипт запускает "чистый" эксперимент на одном компьютере:
# - Server A и Server B работают ПОЛНОСТЬЮ НЕЗАВИСИМО
# - Между ними НЕТ передачи данных
# - Они знают только НАЗВАНИЕ смысла из общего конфига
#
# Использование:
#   ./run_experiment.sh [meaning_name]
#
# Примеры:
#   ./run_experiment.sh bullish_trend
#   ./run_experiment.sh high_volatility
#   ./run_experiment.sh breakout
#
# =============================================================================

set -e  # Exit on error

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Директории
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_A_DIR="$SCRIPT_DIR/server_a"
SERVER_B_DIR="$SCRIPT_DIR/server_b"
DATA_FILE="$SCRIPT_DIR/../data/BTC_USDT_USDT-4h-futures.feather"
OUTPUT_DIR="$SCRIPT_DIR/experiment_results"

# Параметры эксперимента
MEANING_NAME="${1:-bullish_trend}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_ID="${MEANING_NAME}_${TIMESTAMP}"

echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║        DISTRIBUTED SHAMAN EXPERIMENT - LOCAL TEST                ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${YELLOW}Experiment Configuration:${NC}"
echo "  Meaning Name:    $MEANING_NAME"
echo "  Experiment ID:   $EXPERIMENT_ID"
echo "  Data File:       $DATA_FILE"
echo "  Output Dir:      $OUTPUT_DIR/$EXPERIMENT_ID"
echo ""

# Проверка наличия данных
if [ ! -f "$DATA_FILE" ]; then
    echo -e "${RED}ERROR: Data file not found: $DATA_FILE${NC}"
    exit 1
fi

# Создание директории для результатов
mkdir -p "$OUTPUT_DIR/$EXPERIMENT_ID/server_a"
mkdir -p "$OUTPUT_DIR/$EXPERIMENT_ID/server_b"

# =============================================================================
# PHASE 1: Server A (Learner)
# =============================================================================
echo -e "${CYAN}"
echo "┌──────────────────────────────────────────────────────────────────┐"
echo "│  PHASE 1: SERVER A (LEARNER)                                     │"
echo "│  Learning meaning '$MEANING_NAME' from data portion [0.0, 0.5]   │"
echo "└──────────────────────────────────────────────────────────────────┘"
echo -e "${NC}"

SERVER_A_START=$(date +%s.%N)

cd "$SERVER_A_DIR"
python server.py \
    --meaning "$MEANING_NAME" \
    --data "$DATA_FILE" \
    --output "$OUTPUT_DIR/$EXPERIMENT_ID/server_a" \
    --portion-start 0.0 \
    --portion-end 0.5

SERVER_A_END=$(date +%s.%N)
SERVER_A_TIME=$(python3 -c "print(f'{$SERVER_A_END - $SERVER_A_START:.2f}')")

echo -e "${GREEN}Server A completed in ${SERVER_A_TIME}s${NC}"
echo ""

# =============================================================================
# ISOLATION BARRIER - No data transfer!
# =============================================================================
echo -e "${RED}"
echo "════════════════════════════════════════════════════════════════════"
echo "  ⛔  ISOLATION BARRIER - NO DATA TRANSFER BETWEEN SERVERS  ⛔"
echo "  Server B knows ONLY the meaning name: '$MEANING_NAME'"
echo "════════════════════════════════════════════════════════════════════"
echo -e "${NC}"
echo ""

# =============================================================================
# PHASE 2: Server B (Shaman)
# =============================================================================
echo -e "${CYAN}"
echo "┌──────────────────────────────────────────────────────────────────┐"
echo "│  PHASE 2: SERVER B (SHAMAN)                                      │"
echo "│  Searching meaning '$MEANING_NAME' in data portion [0.5, 1.0]    │"
echo "└──────────────────────────────────────────────────────────────────┘"
echo -e "${NC}"

SERVER_B_START=$(date +%s.%N)

cd "$SERVER_B_DIR"
python server.py \
    --meaning "$MEANING_NAME" \
    --data "$DATA_FILE" \
    --output "$OUTPUT_DIR/$EXPERIMENT_ID/server_b" \
    --portion-start 0.5 \
    --portion-end 1.0 \
    --min-score 0.5

SERVER_B_END=$(date +%s.%N)
SERVER_B_TIME=$(python3 -c "print(f'{$SERVER_B_END - $SERVER_B_START:.2f}')")

echo -e "${GREEN}Server B completed in ${SERVER_B_TIME}s${NC}"
echo ""

# =============================================================================
# PHASE 3: Results Analysis
# =============================================================================
echo -e "${CYAN}"
echo "┌──────────────────────────────────────────────────────────────────┐"
echo "│  PHASE 3: EXPERIMENT RESULTS ANALYSIS                            │"
echo "└──────────────────────────────────────────────────────────────────┘"
echo -e "${NC}"

# Читаем результаты Server A
SERVER_A_METRICS="$OUTPUT_DIR/$EXPERIMENT_ID/server_a/training_metrics.json"
if [ -f "$SERVER_A_METRICS" ]; then
    echo -e "${YELLOW}Server A (Learner) Results:${NC}"
    echo "  File: $SERVER_A_METRICS"
    
    A_SAMPLES=$(python3 -c "import json; d=json.load(open('$SERVER_A_METRICS')); print(d['samples_found'])")
    A_SYMBOL_SCORE=$(python3 -c "import json; d=json.load(open('$SERVER_A_METRICS')); print(f\"{d['symbol_match_score']:.4f}\")")
    A_MORPHISM_SCORE=$(python3 -c "import json; d=json.load(open('$SERVER_A_METRICS')); print(f\"{d['morphism_match_score']:.4f}\")")
    A_DISTANCE=$(python3 -c "import json; d=json.load(open('$SERVER_A_METRICS')); print(f\"{d['mean_distance_to_expected']:.4f}\")")
    
    echo "  Samples Found:     $A_SAMPLES"
    echo "  Symbol Match:      $A_SYMBOL_SCORE"
    echo "  Morphism Match:    $A_MORPHISM_SCORE"
    echo "  Mean Distance:     $A_DISTANCE"
    echo ""
fi

# Читаем результаты Server B
SERVER_B_REPORT="$OUTPUT_DIR/$EXPERIMENT_ID/server_b/shaman_report.json"
if [ -f "$SERVER_B_REPORT" ]; then
    echo -e "${YELLOW}Server B (Shaman) Results:${NC}"
    echo "  File: $SERVER_B_REPORT"
    
    B_SUCCESS=$(python3 -c "import json; d=json.load(open('$SERVER_B_REPORT')); print(d['search_successful'])")
    B_BEST_SCORE=$(python3 -c "import json; d=json.load(open('$SERVER_B_REPORT')); print(f\"{d['best_score']:.4f}\")")
    B_MATCHES=$(python3 -c "import json; d=json.load(open('$SERVER_B_REPORT')); print(d['total_matches'])")
    B_WINDOWS=$(python3 -c "import json; d=json.load(open('$SERVER_B_REPORT')); print(d['total_windows_scanned'])")
    B_TIME=$(python3 -c "import json; d=json.load(open('$SERVER_B_REPORT')); print(f\"{d['search_time_seconds']:.2f}\")")
    
    echo "  Search Successful: $B_SUCCESS"
    echo "  Best Score:        $B_BEST_SCORE"
    echo "  Total Matches:     $B_MATCHES / $B_WINDOWS windows"
    echo "  Match Rate:        $(python3 -c "print(f'{$B_MATCHES/$B_WINDOWS*100:.1f}%')")"
    echo "  Search Time:       ${B_TIME}s"
    echo ""
    
    echo -e "${YELLOW}Top 5 Matches (Server B):${NC}"
    python3 -c "
import json
d = json.load(open('$SERVER_B_REPORT'))
for i, r in enumerate(d['top_results'][:5], 1):
    print(f\"  {i}. Window {r['window_index']:5d}: score={r['score']:.4f}, morphisms={r['dominant_morphisms'][:3]}\")
"
    echo ""
fi

# =============================================================================
# PHASE 4: Cross-Comparison
# =============================================================================
echo -e "${CYAN}"
echo "┌──────────────────────────────────────────────────────────────────┐"
echo "│  PHASE 4: CROSS-COMPARISON ANALYSIS                              │"
echo "└──────────────────────────────────────────────────────────────────┘"
echo -e "${NC}"

python3 << EOF
import json

# Load results
try:
    with open('$SERVER_A_METRICS') as f:
        server_a = json.load(f)
    with open('$SERVER_B_REPORT') as f:
        server_b = json.load(f)
except FileNotFoundError as e:
    print(f"Error loading results: {e}")
    exit(1)

print("Meaning: $MEANING_NAME")
print("")

# Comparison metrics
a_score = 1.0 - server_a['mean_distance_to_expected']  # Convert distance to score
b_score = server_b['best_score']

print(f"{'Metric':<30} {'Server A':<15} {'Server B':<15} {'Diff':<10}")
print("-" * 70)
print(f"{'Best Score':<30} {a_score:<15.4f} {b_score:<15.4f} {abs(a_score - b_score):<10.4f}")
print(f"{'Symbol Match':<30} {server_a['symbol_match_score']:<15.4f} {'-':<15} {'-':<10}")
print(f"{'Morphism Match':<30} {server_a['morphism_match_score']:<15.4f} {'-':<15} {'-':<10}")
print(f"{'Samples/Matches':<30} {server_a['samples_found']:<15} {server_b['total_matches']:<15} {'-':<10}")
print("")

# Key insight
score_diff = abs(a_score - b_score)
if score_diff < 0.1:
    result = "✅ EXCELLENT"
    desc = "Servers found very similar patterns independently!"
elif score_diff < 0.2:
    result = "✅ GOOD"
    desc = "Servers found similar patterns with minor differences."
elif score_diff < 0.3:
    result = "⚠️ MODERATE"
    desc = "Some alignment, but notable differences exist."
else:
    result = "❌ POOR"
    desc = "Servers found very different patterns."

print(f"Resonance Quality: {result}")
print(f"Score Difference:  {score_diff:.4f}")
print(f"Interpretation:    {desc}")
print("")

# Success criteria for "shaman" experiment
if server_b['search_successful'] and b_score > 0.6:
    print("🎯 SHAMAN EXPERIMENT: SUCCESS")
    print("   Server B (Shaman) found meaningful patterns matching the concept")
    print("   WITHOUT receiving any data from Server A!")
else:
    print("❓ SHAMAN EXPERIMENT: INCONCLUSIVE")
    print("   More analysis needed.")
EOF

# =============================================================================
# Summary
# =============================================================================
echo ""
echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                    EXPERIMENT SUMMARY                            ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

TOTAL_TIME=$(python3 -c "print(f'{$SERVER_A_TIME + $SERVER_B_TIME:.2f}')")

echo -e "${GREEN}Experiment completed successfully!${NC}"
echo ""
echo "  Experiment ID:     $EXPERIMENT_ID"
echo "  Meaning Tested:    $MEANING_NAME"
echo "  Server A Time:     ${SERVER_A_TIME}s"
echo "  Server B Time:     ${SERVER_B_TIME}s"
echo "  Total Time:        ${TOTAL_TIME}s"
echo ""
echo "  Results saved to:  $OUTPUT_DIR/$EXPERIMENT_ID/"
echo ""
echo -e "${YELLOW}Key Finding:${NC}"
echo "  Both servers searched for '$MEANING_NAME' pattern INDEPENDENTLY,"
echo "  using only shared a priori knowledge (meanings.py)."
echo "  NO data was transferred between them!"
echo ""
