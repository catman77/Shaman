#!/bin/bash
# Run Advanced Shaman Experiment
# Neural Network Consciousness Transfer

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║         ADVANCED SHAMAN EXPERIMENT                               ║"
echo "║         Neural Network Consciousness Transfer                    ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Default parameters
CONSCIOUSNESS="${1:-analytical_professor}"
SKILL="${2:-math_word_problems}"
MODE="${3:-quick}"

echo -e "${YELLOW}Parameters:${NC}"
echo "  Consciousness: $CONSCIOUSNESS"
echo "  Skill: $SKILL"
echo "  Mode: $MODE"
echo ""

# Create output directory
OUTPUT_DIR="./experiment_results/${CONSCIOUSNESS}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Run experiment
if [ "$MODE" == "quick" ]; then
    echo -e "${YELLOW}Running in QUICK mode (for testing)${NC}"
    python experiment.py \
        --consciousness "$CONSCIOUSNESS" \
        --skill "$SKILL" \
        --quick \
        --output "$OUTPUT_DIR"
elif [ "$MODE" == "full" ]; then
    echo -e "${YELLOW}Running in FULL mode${NC}"
    python experiment.py \
        --consciousness "$CONSCIOUSNESS" \
        --skill "$SKILL" \
        --samples 100 \
        --epochs 3 \
        --iterations 50 \
        --output "$OUTPUT_DIR"
else
    echo -e "${YELLOW}Running with default parameters${NC}"
    python experiment.py \
        --consciousness "$CONSCIOUSNESS" \
        --skill "$SKILL" \
        --output "$OUTPUT_DIR"
fi

# Check result
RESULT_FILE="$OUTPUT_DIR/experiment_result.json"

if [ -f "$RESULT_FILE" ]; then
    echo -e "\n${GREEN}Experiment completed!${NC}"
    echo -e "Results saved to: ${BLUE}$OUTPUT_DIR${NC}"
    
    # Parse result
    SUCCESS=$(python3 -c "import json; r=json.load(open('$RESULT_FILE')); print('YES' if r['experiment_success'] else 'NO')")
    RESONANCE=$(python3 -c "import json; r=json.load(open('$RESULT_FILE')); print(f\"{r['resonance_score']:.2%}\")")
    STYLE=$(python3 -c "import json; r=json.load(open('$RESULT_FILE')); print(f\"{r['style_transfer_score']:.2%}\")")
    
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════${NC}"
    echo -e "${BLUE}         EXPERIMENT SUMMARY            ${NC}"
    echo -e "${BLUE}═══════════════════════════════════════${NC}"
    echo -e "  Consciousness: ${YELLOW}$CONSCIOUSNESS${NC}"
    echo -e "  Resonance score: ${YELLOW}$RESONANCE${NC}"
    echo -e "  Style transfer: ${YELLOW}$STYLE${NC}"
    echo -e "  Experiment success: ${GREEN}$SUCCESS${NC}"
    echo -e "${BLUE}═══════════════════════════════════════${NC}"
    
    if [ "$SUCCESS" == "YES" ]; then
        echo -e "\n${GREEN}🎯 CONSCIOUSNESS TRANSFER: SUCCESSFUL!${NC}"
        echo -e "${GREEN}The neural network on Server B has acquired the consciousness${NC}"
        echo -e "${GREEN}of the neural network on Server A using ONLY the meaning name!${NC}"
    else
        echo -e "\n${YELLOW}⚠️  Partial transfer achieved. See results for details.${NC}"
    fi
else
    echo -e "${RED}Error: Result file not found${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}Available consciousness styles:${NC}"
echo "  - analytical_professor"
echo "  - creative_artist"
echo "  - efficient_engineer"
echo "  - wise_mentor"
echo "  - rigorous_scientist"
echo "  - intuitive_guide"
echo ""
echo -e "Usage: ${YELLOW}./run_advanced.sh [consciousness] [skill] [mode]${NC}"
echo "  mode: quick | full | default"
