IN_ROOT="/home/ud202380664/NPJ_Manuscript/ERA5-Land/ERA_LAND/tp_daily_remap"
OUT="/home/ud202380664/NPJ_Manuscript/ERA5-Land/ERA_LAND/ERA5-Land.2000.2022.PRE.daily.CHINA.nc"
TMPDIR="/tmp/cdo_merge_daily"
mkdir -p "${TMPDIR}"

find "${IN_ROOT}" -type f -name "*.nc" | sort > "${TMPDIR}/list.txt"

# 每 200 个文件一组（可按需要改大/改小）
split -l 200 "${TMPDIR}/list.txt" "${TMPDIR}/chunk_"

parts=()
i=0
for c in "${TMPDIR}"/chunk_*; do
  i=$((i+1))
  part="${TMPDIR}/part_${i}.nc"
  echo "[MERGE CHUNK ${i}] -> ${part}"
  cdo -O -L -f nc4 -z zip_4 mergetime $(cat "${c}") "${part}"
  parts+=("${part}")
done

echo "[FINAL MERGE] -> ${OUT}"
cdo -O -L -f nc4 -z zip_4 mergetime "${parts[@]}" "${OUT}"

echo "Done: ${OUT}"
