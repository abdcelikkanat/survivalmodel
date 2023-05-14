
########################################################################################################################
BASEFOLDER="/work3/abce/survival"
PYTHON="/appl/python/3.11.0/bin/python3"
export PYTHONPATH="${PYTHONPATH}:/${BASEFOLDER}"
SCRIPT="${BASEFOLDER}/experiments/split_network.py"
########################################################################################################################
SEED=19
DATASETS=( high-school )
COMPLETION_RATIO=0.0
MASK_RATIO=0.2
PRED_RATIO=0.1
########################################################################################################################
for DATASET in ${DATASETS[@]}
do

INPUT_FILE=${BASEFOLDER}/datasets/real/${DATASET}/${DATASET}.edges
OUTPUT_FOLDER=${BASEFOLDER}/experiments/samples/mr=${MASK_RATIO}_cr=${COMPLETION_RATIO}_pr=${PRED_RATIO}/${DATASET}/

CMD="${PYTHON} ${SCRIPT}"
CMD="${CMD} --edges ${INPUT_FILE} --output_folder ${OUTPUT_FOLDER}"
CMD="${CMD} --pr ${PRED_RATIO} --cr ${COMPLETION_RATIO} --mr ${MASK_RATIO}"
CMD="${CMD} --seed ${SEED}"
${CMD}

done
########################################################################################################################
