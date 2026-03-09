#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
# export CUDA_LAUNCH_BLOCKING=1

echo "🚀 Starting LexCheck Parameter Grid Search..."

# --- Configuration parameter list ---
#"distilbert-base-uncased" "Qwen/Qwen2.5-0.5B_gen" "Qwen/Qwen2.5-0.5B" "facebook/bart-large"
MODELS=("Qwen/Qwen2.5-0.5B_gen") # If multiple models, separate by space, e.g., ("model1" "model2")
EXPLAINERS=("ig" "random" "occlusion" ) #"ig" "random" "occlusion" "attn"
STRATEGIES=("prefix" "lm" "random") # "prefix" "lm"
OPERATIONS=("inject") # "inject" "ablate"
THRESHOLD=0.0
TASK="sst2"

# --- Helper functions ---
run_config() {
    echo "--------------------------------------------------"
    echo "▶️  Executing: $@"
    if python "$@"; then
        echo "✅ SUCCESS"
    else
        echo "❌ FAILED" >&2
        echo "$(date): FAILED $@" >> error_log.txt
    fi
}

# --- Start nested loops ---
for model in "${MODELS[@]}"; do
    for explainer in "${EXPLAINERS[@]}"; do

        # Logic check 1: attn only for distilbert (assume non-distilbert models don't support it)
        if [[ "$explainer" == "attn" && "$model" == *"Qwen"* ]]; then
            echo "⏭️  Skipping: $explainer for Qwen model $model"
            continue
        fi

        for strategy in "${STRATEGIES[@]}"; do
            for operation in "${OPERATIONS[@]}"; do

                # Logic check 2: ablate does not have prefix strategy
                if [[ "$operation" == "ablate" && "$strategy" == "prefix" ]]; then
                    echo "⏭️  Skipping: prefix strategy for ablate operation"
                    continue
                fi

                # Execute experiment
                run_config run_test_generation.py \
                    --task "$TASK" \
                    --model "$model" \
                    --explainer "$explainer" \
                    --strategy "$strategy" \
                    --threshold "$THRESHOLD" \
                    --operation "$operation" \
                    --steps mine,probe


            done
        done
    done
done

echo "--------------------------------------------------"
echo "🏁 All experiments finished. Check error_log.txt for failures."