#!/bin/bash
#SBATCH --job-name=massive_rnn
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=120:00:00
#SBATCH --partition=ml_gpus
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G

# Create logs directory if it doesn't exist
mkdir -p ./logs

# Navigate to project directory
cd /home/v/proj/zebra

# Activate virtual environment
source /home/v/proj/zebra/venv/bin/activate

# Function to intelligently select GPU with the lowest combined memory and GPU utilization
select_optimal_gpu() {
    gpu_info=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits)
    
    best_gpu=-1
    best_score=999999
    
    while IFS=',' read -r gpu_id mem_used mem_total gpu_util; do
        # Trim whitespace
        gpu_id=$(echo "$gpu_id" | xargs)
        mem_used=$(echo "$mem_used" | xargs)
        mem_total=$(echo "$mem_total" | xargs)
        gpu_util=$(echo "$gpu_util" | xargs)
        
        # Avoid division by zero if mem_total is 0 for some reason
        if [ "$mem_total" -eq 0 ]; then
            continue
        fi

        mem_percent=$((mem_used * 100 / mem_total))
        score=$((mem_percent + gpu_util))
        
        if [ "$score" -lt "$best_score" ]; then
            best_score=$score
            best_gpu=$gpu_id
        fi
    done <<< "$gpu_info"
    
    echo $best_gpu
}

# Select GPU
if [ -n "$REQUESTED_GPU" ]; then
    export CUDA_VISIBLE_DEVICES=$REQUESTED_GPU
else
    selected_gpu=$(select_optimal_gpu)
    if [ "$selected_gpu" -eq -1 ]; then
        echo "Error: Could not find an available GPU." >&2
        exit 1
    fi
    export CUDA_VISIBLE_DEVICES=$selected_gpu
fi

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:/home/v/proj/zebra"

# W&B configuration (uncomment and set your entity if using W&B)
export WANDB_PROJECT="zebra-neuron-forecasting"
# export WANDB_ENTITY="your_entity"
export WANDB_NAME="${WANDB_RUN_NAME:-"massive-rnn-$SLURM_JOB_ID"}"
export WANDB_TAGS="gpu${CUDA_VISIBLE_DEVICES}"

# Print summary
echo "--- Job Summary ---"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Running on GPU: $CUDA_VISIBLE_DEVICES"
echo "Working Directory: $(pwd)"
echo "W&B Project: $WANDB_PROJECT"
echo "W&B Run Name: $WANDB_NAME"
echo "-------------------"

# Run the training script
echo "Starting training at $(date)..."

# Default command - modify parameters as needed via --params
python massive_rnn_train.py --params sequence_length=32 batch_size=32 accumulate_grad_batches=1 \
       connectivity_path="connectivity_graph_global_threshold.pkl" min_connection_strength=0.52 \
       hidden_dim=9 \
       strategy=auto precision="32" use_gradient_checkpointing=False

if [ $? -eq 0 ]; then
    echo "Training completed successfully at $(date)"
else
    echo "Training failed at $(date)"
    exit 1
fi

echo "Job completed at $(date)"
