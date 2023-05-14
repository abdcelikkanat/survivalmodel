
########################################################################################################################
nvidia-smi
# Load the cuda module
module load cuda/10.2
/appl/cuda/10.2/samples/NVIDIA_CUDA-10.2_Samples/bin/x86_64/linux/release/deviceQuery
########################################################################################################################
BASEFOLDER="/work3/abce/survival"
PYTHON="/appl/python/3.11.0/bin/python3"
export PYTHONPATH="${PYTHONPATH}:/${BASEFOLDER}"
SCRIPT="${BASEFOLDER}/run.py"
########################################################################################################################
DATASETS=( high-school )
BIN=100
K=25
LAMBDA=1e6
DIM=2
EPOCH=100
SPE=10
BATCH_SIZE=100
LR=0.1
SEED=19
########################################################################################################################
INPUT_FOLDER=${BASEFOLDER}/datasets/real/
MODEL_FOLDER=${BASEFOLDER}/experiments/models_single_runs/
LOG_FOLDER=${BASEFOLDER}/experiments/logs_single_runs/
mkdir ${MODEL_FOLDER}
mkdir ${LOG_FOLDER}
########################################################################################################################
for DATASET in ${DATASETS[@]}
do

# Print Information
echo "Dataset="${DATASET} "Lambda="${LAMBDA}
# Define model name
MODELNAME="single_run_${DATASET}_B=${BIN}_K=${K}_lambda=${LAMBDA}_dim=${DIM}"
MODELNAME="${MODELNAME}_epoch=${EPOCH}_spe=${SPE}_bs=${BATCH_SIZE}_lr=${LR}_seed=${SEED}"
# Define input, output and log path
INPUT_PATH=${INPUT_FOLDER}/${DATASET}/${DATASET}.edges
MODEL_PATH=${MODEL_FOLDER}/${MODELNAME}.model
LOG=${LOG_FOLDER}/${MODELNAME}.txt
# Define the command
CMD="${PYTHON} ${SCRIPT} --edges ${INPUT_PATH} --model_path ${MODEL_PATH}"
CMD="${CMD} --log ${LOG} --dim ${DIM} --bins_num ${BIN} --k ${K} --batch_size ${BATCH_SIZE}"
CMD="${CMD} --prior_lambda ${LAMBDA} --epoch_num ${EPOCH} --lr ${LR} --seed ${SEED} --spe ${SPE} --verbose 1"

$CMD

done

########################################################################################################################
