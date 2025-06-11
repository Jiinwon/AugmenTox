#!/bin/bash

# ======== 백그라운드 실행 래퍼 시작 ========
if [ -z "$LAUNCHED" ]; then
    export LAUNCHED=1
    LOG_FILE="$(pwd)/launcher_stat_$(date +%Y%m%d_%H%M%S).log"
    nohup bash "$0" > "$LOG_FILE" 2>&1 &
    echo "백그라운드로 실행되었습니다. 로그: $LOG_FILE"
    exit
fi
# ======== 백그라운드 실행 래퍼 끝 ========

#SBATCH --job-name=STAT_launcher
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --cpus-per-task=1
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -e

export MKL_THREADING_LAYER=GNU

# 스크립트 실행 위치(프로젝트 루트)
REPO_DIR="$(pwd)"

# Slurm에서 동시에 실행할 최대 작업 수
MAX_SLURM_JOBS=20
PARTITIONS=(gpu1 gpu2 gpu3 gpu4 gpu5 gpu6)
DEFAULT_PART="gpu1"
GRES="gpu"
MAX_RUNNING_PER_PART=10


function running_in_partition {
    local part="$1"
    squeue -u "$USER" -p "$part" -t R -h | wc -l
}

function has_idle_nodes {
    local part="$1"
    sinfo -h -p "$part" -o "%D %t" \
        | awk '$2=="idle" || $2=="mix" {print $1}' \
        | grep -q '[1-9]'
}

function current_slurm_jobs {
    squeue -u "$USER" -h | wc -l
}

function wait_for_slot {
    while [ "$(current_slurm_jobs)" -ge "$MAX_SLURM_JOBS" ]; do
        sleep 60
    done
}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR"

# 조합 목록은 visual/performance_summary.xlsx의 'p-value' 시트에서 추출한다.
readarray -t COMBOS < <(python3 "$SCRIPT_DIR/statistics/get_combinations_from_excel.py")

DATE=$(date +"%Y%m%d")
TIME=$(date +"%H%M")
BASE_LOG="./log/${DATE}/${TIME}_stat"
mkdir -p "$BASE_LOG"

for combo in "${COMBOS[@]}"; do
    IFS=$'\t' read -r SRC TGT MODEL <<< "$combo"
    PAIR="${SRC}&&${TGT}_${MODEL}"
    OUT="${BASE_LOG}/${PAIR}.out"
    ERR="${BASE_LOG}/${PAIR}.err"

        wait_for_slot
        SUBMITTED=0

        if [ "$(current_slurm_jobs)" -lt "$MAX_SLURM_JOBS" ]; then
            for p in "${PARTITIONS[@]}"; do
                if [ "$(running_in_partition "$p")" -lt "$MAX_RUNNING_PER_PART" ] && has_idle_nodes "$p"; then
                    JOBID=$(sbatch --parsable --job-name="STAT_${SRC}_${TGT}" \
                                   --partition="$p" \
                                   --gres="${GRES}" \
                                   --cpus-per-task=8 \
                                   --output="$OUT" \
                                   --error="$ERR" \
                                   --export=ALL,REPO_DIR="$REPO_DIR",SOURCE_NAME="${SRC}",TARGET_NAME="${TGT}",MODEL_TYPE="$MODEL",LOG_SUBDIR="$BASE_LOG" \
                                   "${SCRIPT_DIR}/statistics/run_single_statistics.sh")
                    sleep 3
                    STATE=$(squeue -j "$JOBID" -h -o "%T")
                    if [ "$STATE" = "RUNNING" ]; then
                        SUBMITTED=1
                        break
                    else
                        scancel "$JOBID"
                    fi
                fi
            done

            if [ "$SUBMITTED" -eq 0 ] && [ "$(current_slurm_jobs)" -lt "$MAX_SLURM_JOBS" ]; then
                sbatch --job-name="STAT_${SRC}_${TGT}" \
                       --partition="${DEFAULT_PART}" \
                       --gres="${GRES}" \
                       --cpus-per-task=8 \
                       --output="$OUT" \
                       --error="$ERR" \
                       --export=ALL,REPO_DIR="$REPO_DIR",SOURCE_NAME="${SRC}",TARGET_NAME="${TGT}",MODEL_TYPE="$MODEL",LOG_SUBDIR="$BASE_LOG" \
                       "${SCRIPT_DIR}/statistics/run_single_statistics.sh"
                SUBMITTED=1
            fi
        fi

done


echo "✅ 모든 통계 검정 실행 명령 전송 완료"

