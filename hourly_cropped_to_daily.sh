#!/usr/bin/env bash
set -euo pipefail

# 裁剪/重网格后的小时数据根目录
IN_ROOT="/home/ud202380664/NPJ_Manuscript/ERA5-Land/ERA_LAND/tp_hourly_monthly_remap"
# 逐日输出根目录
OUT_ROOT="/home/ud202380664/NPJ_Manuscript/ERA5-Land/ERA_LAND/tp_daily_remap"

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
    echo "[INFO] ${IN_DIR} 下无 nc，跳过。"
    continue
  fi

  for f in "${files[@]}"; do
    base="$(basename "$f" .nc)"
    out="${OUT_DIR}/${base}.daily.nc"

    if [ -s "${out}" ]; then
      echo "[SKIP] 已存在：${out}"
      continue
    fi

    # 读取单位（只取 tp 变量的单位）
    unit="$(cdo -s showunit -selname,tp "${f}" 2>/dev/null | tr -d '[:space:]' || true)"
    echo "[DAILY] ${f} -> ${out} (unit=${unit})"

    # 对 tp：daysum 得到日累计
    # 若单位为 m，则转 mm（乘1000），并设单位为 mm
    if [[ "${unit}" == "m" || "${unit}" == "meter" || "${unit}" == "meters" ]]; then
      cdo -O -L -P "${NPROC}" -f nc4 -z zip_4 \
        -setname,pr -setunit,"mm" -mulc,1000 -daysum \
        -selname,tp "${f}" "${out}"
    else
      # 若已经是 mm 或 kg m-2 等（等价 mm），直接 daysum
      cdo -O -L -P "${NPROC}" -f nc4 -z zip_4 \
        -setname,pr -daysum \
        -selname,tp "${f}" "${out}"
    fi
  done
done

echo "Done."
