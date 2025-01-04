#!/bin/bash

# Set the base directory for logs
LOG_DIR="logs"
SCRIPT="main.py"  # Replace with the path to your Python script
DATASET="tren_data2"            # Replace with your dataset name

# Ensure the log directory exists
mkdir -p "$LOG_DIR"

# Define parameter grids
OPTIMIZERS=("sgd" "adam")
ACTIVATIONS=("relu" "sigmoid")
LAYERS=(1 2 4 8 16)
NEURONS=(4 8 16 32 64 128)
LEARNING_RATES=(0.01 0.001 0.0001)

# Parallel execution settings
MAX_PROCESSES=32
CURRENT_PROCESSES=0

# Spočítáme celkový počet kombinací
TOTAL_TASKS=0
for OPTIMIZER in "${OPTIMIZERS[@]}"; do
    for ACTIVATION in "${ACTIVATIONS[@]}"; do
        for NUM_LAYERS in "${LAYERS[@]}"; do
            for NUM_NEURONS in "${NEURONS[@]}"; do
                for LEARNING_RATE in "${LEARNING_RATES[@]}"; do
                    TOTAL_TASKS=$((TOTAL_TASKS + 1))
                done
            done
        done
    done
done

COMPLETED_TASKS=0

# Funkce pro zobrazení progress baru
print_progress() {
    # Vypočítáme procenta
    local progress=$((100*COMPLETED_TASKS/TOTAL_TASKS))
    printf "\rProgress: %3d%% (%d/%d)" "$progress" "$COMPLETED_TASKS" "$TOTAL_TASKS"
}

# Iterate over all combinations of parameters
for OPTIMIZER in "${OPTIMIZERS[@]}"; do
    for ACTIVATION in "${ACTIVATIONS[@]}"; do
        for NUM_LAYERS in "${LAYERS[@]}"; do
            for NUM_NEURONS in "${NEURONS[@]}"; do
                for LEARNING_RATE in "${LEARNING_RATES[@]}"; do

                    # Create a unique name for the run
                    RUN_NAME="${OPTIMIZER}_${ACTIVATION}_${NUM_LAYERS}layers_${NUM_NEURONS}neurons_lr${LEARNING_RATE}"
                    RUN_LOG_DIR="${LOG_DIR}/${RUN_NAME}"

                    # Create a log directory for this run
                    mkdir -p "$RUN_LOG_DIR"

                    # Run the script in the background
                    python "$SCRIPT" \
                        --optimizer "$OPTIMIZER" \
                        --activation "$ACTIVATION" \
                        --num_layers "$NUM_LAYERS" \
                        --layer_width "$NUM_NEURONS" \
                        --learning_rate "$LEARNING_RATE" \
                        --log_dir "$RUN_LOG_DIR" \
                        --dataset "$DATASET" \
                        --name "$RUN_NAME" > "${RUN_LOG_DIR}/output.log" 2>&1 &

                    # Increment the process counter
                    CURRENT_PROCESSES=$((CURRENT_PROCESSES + 1))

                    # Pokud jsme dosáhli maxima, počkáme na uvolnění
                    if [ "$CURRENT_PROCESSES" -ge "$MAX_PROCESSES" ]; then
                        wait -n
                        CURRENT_PROCESSES=$((CURRENT_PROCESSES - 1))
                        COMPLETED_TASKS=$((COMPLETED_TASKS + 1))
                        print_progress
                    fi
                done
            done
        done
    done
done

# Po spuštění všech úloh nyní čekáme na jejich dokončení
while [ "$CURRENT_PROCESSES" -gt 0 ]; do
    wait -n
    CURRENT_PROCESSES=$((CURRENT_PROCESSES - 1))
    COMPLETED_TASKS=$((COMPLETED_TASKS + 1))
    print_progress
done

echo
echo "All runs have completed."
