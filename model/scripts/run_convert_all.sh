#!/usr/bin/env bash
# run_convert_all.sh

# 1) config.py 에 정의된 SOURCE_NAMES, TARGET_NAMES 읽기
readarray -t SOURCE_NAMES < <(
  python3 - << 'PYCODE'
import config.config as cfg
for name in cfg.SOURCE_NAMES:
    print(name)
PYCODE
)

readarray -t TARGET_NAMES < <(
  python3 - << 'PYCODE'
import config.config as cfg
for name in cfg.TARGET_NAMES:
    print(name)
PYCODE
)


python -m scripts.convert_excel_to_csv


