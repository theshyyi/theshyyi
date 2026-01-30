#!/usr/bin/env bash
set -euo pipefail

# 输入根目录（2000/ ... 2022/）
IN_ROOT="/home/ud202380664/NPJ_Manuscript/ERA5-Land/ERA_LAND/tp_hourly_monthly"
# 输出根目录（你自己定）
OUT_ROOT="/home/ud202380664/NPJ_Manuscript/ERA5-Land/ERA_LAND/tp_hourly_monthly_remap"
# 目标网格描述文件（你提供的 txt）
GRID_TXT="/home/ud202380664/PRE_MERGE/code/grid_0p25_lat50.txt"

# 选择一种 remap 方法：remapbil / remapcon / remapnn
REMAP_METHOD="remapbil"

# 并行线程（CDO -P）
NPROC=8

mkdir -p "${OUT_ROOT}"

for Y in $(seq 2000 2022); do
  IN_DIR="${IN_ROOT}/${Y}"
  OUT_DIR="${OUT_ROOT}/${Y}"
  mkdir -p "${OUT_DIR}"

  shopt -s nullglob
  files=("${IN_DIR}"/*.nc)
  shopt -u nullglob

  if [ ${#files[@]} -eq 0 ]; then
    echo "[INFO] ${IN_DIR} 下没有 nc 文件，跳过。"
    continue
  fi

  for f in "${files[@]}"; do
    base="$(basename "$f")"
    out="${OUT_DIR}/${base}"

    if [ -s "${out}" ]; then
      echo "[SKIP] 已存在：${out}"
      continue
    fi

    echo "[DO] ${f} -> ${out}"
    # -L：大文件更稳；-f nc4 + -z：压缩输出；-O：覆盖（这里也会被跳过逻辑保护）
    cdo -O -L -P "${NPROC}" -f nc4 -z zip_4 \
      ${REMAP_METHOD},"${GRID_TXT}" \
      "${f}" "${out}"
  done
done

echo "Done."
