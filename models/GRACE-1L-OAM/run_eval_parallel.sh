#!/bin/bash

## this script is for imitation of SLURM tasks array on local machine with multiple GPUs

THREADS=8  # number of threads for OMP, MKL, NUMEXPR etc. To share resources on single machine

GPU_LIST=(-1 -1 -1 -1)  # Define the list of GPUs to use
NGPU=${#GPU_LIST[@]}  # Set NGPU to the number of GPUs in GPU_LIST, automatically determined from GPU_LIST

model_name="GRACE" # just for information, model name is hardcoded in 1_test_srme.py

export MODEL_NAME="${model_name}"
echo "MODEL_NAME=${MODEL_NAME}"
export MKL_NUM_THREADS=$THREADS
export NUMEXPR_NUM_THREADS=$THREADS
export OMP_NUM_THREADS=$THREADS

export SLURM_ARRAY_TASK_COUNT=16  # slurm_array_task_count
export SLURM_ARRAY_JOB_ID="production"
export SLURM_JOB_ID="production"


# Function to kill all child processes when the script receives a termination signal
cleanup() {
  echo "Terminating all child processes..."
  pkill -P $$
  exit 1
}

# Set trap to catch signals and trigger the cleanup function
trap cleanup SIGINT SIGTERM


# skip1 and skip2 are options to control execution
# of first and second stages of workflow

# Check if "skip1" is in the arguments
skip1=false
for arg in "$@"; do
  if [ "$arg" = "skip1" ]; then
    skip1=true
    break
  fi
done

skip2=false
for arg in "$@"; do
  if [ "$arg" = "skip2" ]; then
    skip2=true
    break
  fi
done

exec > "${MODEL_NAME}.out" 2>&1
echo "MODEL_NAME=${MODEL_NAME}"
echo "SLURM_ARRAY_TASK_COUNT=${SLURM_ARRAY_TASK_COUNT}"

if [ "$skip1" = false ]; then
  echo "Running 1_test_srme.py"
  for task_id in $(seq 0 $((SLURM_ARRAY_TASK_COUNT-1)))
  do
    CUDA_VISIBLE_DEVICES=${GPU_LIST[$((task_id % NGPU))]} SLURM_ARRAY_TASK_ID=${task_id}  python 1_test_srme.py "$MODEL_NAME" &
  done
  wait
else
  echo "Skip 1_force_sets"
fi

if [ "$skip2" = false ]; then
  echo "Running 2_evaluate"
  for task_id in $(seq 0 $((SLURM_ARRAY_TASK_COUNT-1)))
  do
    SLURM_ARRAY_TASK_ID=${task_id} python 2_evaluate.py "$MODEL_NAME" &
  done
  wait
else
  echo "Skip 2_evaluate"
fi
