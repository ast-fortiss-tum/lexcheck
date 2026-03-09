#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
#export CUDA_LAUNCH_BLOCKING=1

echo "🚀 Starting LexCheck Foundation Training Loop..."


run_config() {
    echo "--------------------------------------------------"
    echo "▶️  Executing: $@"
    if python "$@"; then
        echo "✅ SUCCESS: $@"
    else
        echo "❌ FAILED: $@" >&2
        echo "$(date): FAILED $@" >> error_log.txt
    fi
}






# SST
run_config setup.py --tasks sst2 --models distilbert-base-uncased --xai ig,occlusion,attn --max-items 2000
run_config setup.py --tasks sst2 --models Qwen/Qwen2.5-0.5B --xai ig,occlusion --max-items 2000
run_config setup.py --tasks sst2 --models Qwen/Qwen2.5-0.5B --generative --xai ig,occlusion --max-items 2000
run_config setup.py --tasks sst2 --models facebook/bart-large --xai ig,occlusion --max-items 2000

run_config setup.py --tasks news --models distilbert-base-uncased --xai ig,occlusion,attn --max-items 2000
# QNLI (Generative)
run_config setup.py --tasks news --models Qwen/Qwen2.5-0.5B --xai ig,occlusion --max-items 2000
# News (Generative)
#run_config setup.py --tasks news --models Qwen/Qwen2.5-0.5B --generative --xai ig,occlusion --max-items 2000


run_config setup.py --tasks github --models distilbert-base-uncased --xai ig,occlusion,attn --max-items 2000
# QNLI (Generative)
run_config setup.py --tasks github --models Qwen/Qwen2.5-0.5B --xai ig,occlusion --max-items 2000
# News (Generative)
# run_config setup.py --tasks github --models Qwen/Qwen2.5-0.5B --generative --max-items 2000

echo "--------------------------------------------------"
echo "🏁 Training Loop Finished. Check error_log.txt for any failures."