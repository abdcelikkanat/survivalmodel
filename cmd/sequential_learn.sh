
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
MASK_RATIO=0.2
COMPLETION_RATIO=0.1
PRED_RATIO=0.0
########################################################################################################################
BIN=100
K=25
LAMBDA_LIST=( 1e6 1e5 1e4 1e3 1e2 1e1 1e0 )
DIM=2
EPOCH=100
SPE=10
BATCH_SIZE=100
LR=0.1
SEED=19
########################################################################################################################
INPUT_FOLDER=${BASEFOLDER}/experiments/samples/mr=${MASK_RATIO}_cr=${COMPLETION_RATIO}_pr=${PRED_RATIO}/
MODEL_FOLDER=${BASEFOLDER}/experiments/models_mr=${MASK_RATIO}_cr=${COMPLETION_RATIO}_pr=${PRED_RATIO}/
LOG_FOLDER=${BASEFOLDER}/experiments/logs_mr=${MASK_RATIO}_cr=${COMPLETION_RATIO}_pr=${PRED_RATIO}/
mkdir ${MODEL_FOLDER}
mkdir ${LOG_FOLDER}
########################################################################################################################
for DATASET in ${DATASETS[@]}
do

LEN=${#LAMBDA_LIST[@]}
for (( IDX=0; IDX<${LEN}; IDX++ ));
do
# Print Information
echo "Dataset="${DATASET} "Lambda="${LAMBDA_LIST[${IDX}]}
# Define model name
MODELNAME="dec${IDX}_${DATASET}_B=${BIN}_K=${K}_lambda=${LAMBDA_LIST[${IDX}]}_dim=${DIM}"
MODELNAME="${MODELNAME}_epoch=${EPOCH}_spe=${SPE}_bs=${BATCH_SIZE}_lr=${LR}_seed=${SEED}"
# Define input, output and log path
INPUT_PATH=${INPUT_FOLDER}/${DATASET}/train.edges
MODEL_PATH=${MODEL_FOLDER}/${MODELNAME}.model
LOG=${LOG_FOLDER}/${MODELNAME}.txt
# Define the command
CMD="${PYTHON} ${SCRIPT} --edges ${INPUT_PATH} --model_path ${MODEL_PATH}"
CMD="${CMD} --log ${LOG} --dim ${DIM} --bins_num ${BIN} --k ${K} --batch_size ${BATCH_SIZE}"
CMD="${CMD} --prior_lambda ${LAMBDA_LIST[${IDX}]} --epoch_num ${EPOCH} --lr ${LR} --seed ${SEED} --spe ${SPE} --verbose 1"

if [ ${IDX} -gt 0 ]
then
let PREV_IDX=${IDX}-1
PREV_MODELNAME="dec${IDX}_${DATASET}_B=${BIN}_K=${K}_lambda=${LAMBDA_LIST[${PREV_IDX}]}_dim=${DIM}"
PREV_MODELNAME="${PREV_MODELNAME}_epoch=${EPOCH}_spe=${SPE}_bs=${BATCH_SIZE}_lr=${LR}_seed=${SEED}"
INIT_MODEL="${MODEL_FOLDER}/${PREV_MODELNAME}.model"
CMD=${CMD}" --init_model ${INIT_MODEL}"
fi

$CMD

done
done

########################################################################################################################
