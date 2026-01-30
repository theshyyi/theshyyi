in=/home/ud202380664/PRE_MERGE/TIMEFIX/Finish/PERSIANN-UNet.TIMEFIX.daily.CHINA.nc
out=/home/ud202380664/PRE_MERGE/TIMEFIX/PERSIANN-UNet.TIMEFIX.daily.CHINA.FIXED.nc
ref=/home/ud202380664/PRE_MERGE/TIMEFIX/Finish/PERSIANN-CDR-UNet.TIMEFIX.daily.CHINA.nc

var=$(cdo -s showname "$in" | awk '{print $1}')
echo "Detected var = $var"

tmpdir=$(mktemp -d)
trap "rm -rf $tmpdir" EXIT

# 取一个相邻日做模板（2002-04-26 这天一定存在）
cdo -O -L -seldate,2002-04-26,2002-04-26 "$in" "$tmpdir/template.nc"

# 把模板这一天全赋值为 1e20，并将 1e20 设为缺测值（missing）
cdo -O -L -setmissval,1e20 -expr,"$var=1e20" "$tmpdir/template.nc" "$tmpdir/missing.nc"

# 把日期改成 2002-04-27（时间保持 00:00:00）
cdo -O -L -setdate,2002-04-27 "$tmpdir/missing.nc" "$tmpdir/20020427_missing.nc"

# 合并并按时间排序（避免时间轴乱序）
cdo -O -L -mergetime "$in" "$tmpdir/20020427_missing.nc" "$tmpdir/merged.nc"
cdo -O -L -z zip_4 sorttaxis "$tmpdir/merged.nc" "$out"


echo "Wrote: $out"
