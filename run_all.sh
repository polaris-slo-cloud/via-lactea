#!/bin/bash

DATASETS=("wildfire" "floodnet" "c10")
TOPO_SIZES=("250")
#TOPO_SIZES=("250" "500" "1000" "2000" "3000" "6000")

LOG_FILE="run_all_$(date +%Y%m%d_%H%M%S).log"

echo "===== VIA LACTEA FULL RUN =====" | tee -a $LOG_FILE
echo "Started at: $(date)" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

TOTAL_START=$(date +%s)

for DATASET in "${DATASETS[@]}"; do
  for SIZE in "${TOPO_SIZES[@]}"; do

    TOPO="resources/topo_${SIZE}.json"

    echo "----------------------------------------" | tee -a $LOG_FILE
    echo "Dataset: $DATASET | Topology: $SIZE" | tee -a $LOG_FILE
    echo "Started: $(date)" | tee -a $LOG_FILE

    START=$(date +%s)

    python3 main.py --dataset "$DATASET" --topo "$TOPO" >> $LOG_FILE 2>&1

    END=$(date +%s)
    DURATION=$((END - START))

    echo "Finished: $(date)" | tee -a $LOG_FILE
    echo "Duration: ${DURATION} seconds" | tee -a $LOG_FILE
    echo "" | tee -a $LOG_FILE

  done
done

TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - TOTAL_START))

echo "========================================" | tee -a $LOG_FILE
echo "All runs finished at: $(date)" | tee -a $LOG_FILE
echo "Total duration: ${TOTAL_DURATION} seconds" | tee -a $LOG_FILE
echo "Log saved to: $LOG_FILE" | tee -a $LOG_FILE