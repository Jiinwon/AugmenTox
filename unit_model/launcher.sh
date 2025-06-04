#!/bin/bash

# ======== 백그라운드 실행 래퍼 시작 ========
# LAUNCHED 변수가 없으면, 이 스크립트를 nohup 백그라운드로 재실행하고 종료
if [ -z "$LAUNCHED" ]; then
    export LAUNCHED=1
    # 로그 파일 경로 (원하는 위치/이름으로 수정 가능)
    LOG_FILE="$(pwd)/launcher_$(date +%Y%m%d_%H%M%S).log"
    nohup bash "$0" > "$LOG_FILE" 2>&1 &
    echo "백그라운드로 실행되었습니다. 로그: $LOG_FILE"
    exit
fi
# ======== 백그라운드 실행 래퍼 끝 ========



#SBATCH --job-name=GNN_launcher
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --cpus-per-task=1
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -e

# 최대 동시 Slurm 작업 개수
MAX_JOBS=20
# PARTITIONS 선언 위치 (기존 맨 위 부근)
PARTITIONS=(gpu1 gpu2 gpu3 gpu4 gpu5 gpu6)
DEFAULT_PART="gpu1"
GRES="gpu" #"gpu:rtx3090:1"
MAX_RUNNING_PER_PART=10



# “실행(run) 중인 잡” 개수를 특정 파티션에서 세어 반환
function running_in_partition {
    local part="$1"
    squeue -u "$USER" -p "$part" -t R -h | wc -l
}

# 해당 파티션에 유휴(idle) 노드가 있는지 확인 (idle 노드 수 > 0)
function has_idle_nodes {
    local part="$1"
    sinfo -h -p "$part" -o "%D %t" \
        | awk '$2=="idle" || $2=="mix" {print $1}' \
        | grep -q '[1-9]'
}

# 현재 Slurm 작업 개수를 반환
function current_job_count {
    squeue -u "$USER" -h | wc -l
}

# 슬롯이 생길 때까지 대기
function wait_for_slot {
    while [ "$(current_job_count)" -ge "$MAX_JOBS" ]; do
        sleep 60
    done
}

# 현재 스크립트 경로
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR"

# config.py에서 OPERA 읽기
OPERA=$(python3 - << 'PYCODE'
import config.config as cfg
print(cfg.OPERA)
PYCODE
)

# 로그 디렉토리 생성
DATE=$(date +"%Y%m%d")
TIME=$(date +"%H%M")
BASE_LOG="./log/${DATE}/${TIME}"
mkdir -p "${BASE_LOG}"

# config.py에서 SOURCE_NAMES, TARGET_NAMES, MODEL_TYPE 읽기
readarray -t SOURCE_NAMES < <(python3 - << 'PYCODE'
import config.config as cfg
for n in cfg.SOURCE_NAMES: print(n)
PYCODE
)
readarray -t TARGET_NAMES < <(python3 - << 'PYCODE'
import config.config as cfg
for n in cfg.TARGET_NAMES: print(n)
PYCODE
)
MODEL_TYPE=$(python3 - << 'PYCODE'
import config.config as cfg
print(cfg.MODEL_TYPE)
PYCODE
)

# 길이 체크
if [ "${#SOURCE_NAMES[@]}" -ne "${#TARGET_NAMES[@]}" ]; then
  echo "SOURCE_NAMES와 TARGET_NAMES의 길이가 다릅니다." >&2
  exit 1
fi

# OPERA가 True인 경우 SDF 파일을 사용하여 단일 SOURCE_NAME과 모든 TARGET_NAME 조합 수행
if [ "$OPERA" = "True" ]; then
    echo "OPERA 데이터로 pretraining을 수행합니다."
    # config에서 SDF 경로 읽기
    SDF_PATH=$(python3 - << 'PYCODE'
import config.config as cfg
print(cfg.SOURCE_SDF_PATH)
PYCODE
    )
    # SDF 파일명을 SRC 변수로 사용 (확장자 제거)
    SRC=$(basename "$SDF_PATH" .sdf)

    

    for TGT in "${TARGET_NAMES[@]}"; do
        PAIR="${SRC}&&${TGT}_${MODEL_TYPE}"
        OUT="${BASE_LOG}/${PAIR}.out"
        ERR="${BASE_LOG}/${PAIR}.err"
        wait_for_slot
        # PARTITON 정의는 sbatch 직전에 동적으로 결정 → 실행(RUN)될 때까지 순회
        SUBMITTED=0
        for p in "${PARTITIONS[@]}"; do
            if [ "$(running_in_partition "$p")" -lt "$MAX_RUNNING_PER_PART" ] && has_idle_nodes "$p" ]; then
                JOBID=$(sbatch --parsable --job-name="GNN_${SRC}_${TGT}" \
                               --partition="$p" \
                               --gres="${GRES}" \
                               --cpus-per-task=8 \
                               --output="${OUT}" \
                               --error="${ERR}" \
                               --export=ALL,SOURCE_NAME="${SRC}",TARGET_NAME="${TGT}",MODEL_TYPE="$MODEL_TYPE",LOG_SUBDIR="${BASE_LOG}",OPERA="${OPERA}" \
                               "${SCRIPT_DIR}/run_single_pipeline.sh")
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
        if [ "$SUBMITTED" -eq 0 ]; then
            sbatch --job-name="GNN_${SRC}_${TGT}" \
                   --partition="${DEFAULT_PART}" \
                   --gres="${GRES}" \
                   --cpus-per-task=8 \
                   --output="${OUT}" \
                   --error="${ERR}" \
                   --export=ALL,SOURCE_NAME="${SRC}",TARGET_NAME="${TGT}",MODEL_TYPE="$MODEL_TYPE",LOG_SUBDIR="${BASE_LOG}",OPERA="${OPERA}" \
                   "${SCRIPT_DIR}/run_single_pipeline.sh"
        fi
    done
else
    # 기존 방식: SOURCE_NAMES와 TARGET_NAMES의 모든 조합 수행
    for SRC in "${SOURCE_NAMES[@]}"; do
        for TGT in "${TARGET_NAMES[@]}"; do
            PAIR="${SRC}&&${TGT}_${MODEL_TYPE}"
            OUT="${BASE_LOG}/${PAIR}.out"
            ERR="${BASE_LOG}/${PAIR}.err"
            wait_for_slot
            # PARTITON 정의는 sbatch 직전에 동적으로 결정 → 실행(RUN)될 때까지 순회
            SUBMITTED=0
            for p in "${PARTITIONS[@]}"; do
                if [ "$(running_in_partition "$p")" -lt "$MAX_RUNNING_PER_PART" ] && has_idle_nodes "$p" ]; then
                    JOBID=$(sbatch --parsable --job-name="GNN_${SRC}_${TGT}" \
                                   --partition="$p" \
                                   --gres="${GRES}" \
                                   --cpus-per-task=8 \
                                   --output="${OUT}" \
                                   --error="${ERR}" \
                                   --export=ALL,SOURCE_NAME="${SRC}",TARGET_NAME="${TGT}",MODEL_TYPE="${MODEL_TYPE}",LOG_SUBDIR="${BASE_LOG}",OPERA="${OPERA}" \
                                   "${SCRIPT_DIR}/run_single_pipeline.sh")
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
            if [ "$SUBMITTED" -eq 0 ]; then
                sbatch --job-name="GNN_${SRC}_${TGT}" \
                       --partition="${DEFAULT_PART}" \
                       --gres="${GRES}" \
                       --cpus-per-task=8 \
                       --output="${OUT}" \
                       --error="${ERR}" \
                       --export=ALL,SOURCE_NAME="${SRC}",TARGET_NAME="${TGT}",MODEL_TYPE="${MODEL_TYPE}",LOG_SUBDIR="${BASE_LOG}",OPERA="${OPERA}" \
                       "${SCRIPT_DIR}/run_single_pipeline.sh"
            fi
        done
    done
fi
