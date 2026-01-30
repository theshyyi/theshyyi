#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import xarray as xr

def time_as_str(ds):
    if "time" not in ds:
        raise KeyError("No 'time' coordinate found.")
    t = ds["time"].values
    # 对 numpy datetime64 / cftime 都稳：直接转字符串
    return [str(x) for x in t]

def main(f1, f2):
    with xr.open_dataset(f1, decode_times=True, use_cftime=True) as ds1, \
         xr.open_dataset(f2, decode_times=True, use_cftime=True) as ds2:

        t1 = time_as_str(ds1)
        t2 = time_as_str(ds2)

    s1, s2 = set(t1), set(t2)
    extra_in_2 = sorted(s2 - s1)
    extra_in_1 = sorted(s1 - s2)

    print(f"[{f1}] ntime = {len(t1)}")
    print(f"[{f2}] ntime = {len(t2)}")

    if extra_in_2:
        print(f"\nTime points in {f2} but NOT in {f1}:")
        for x in extra_in_2:
            print("  ", x)

    if extra_in_1:
        print(f"\nTime points in {f1} but NOT in {f2}:")
        for x in extra_in_1:
            print("  ", x)

    if not extra_in_1 and not extra_in_2:
        print("\nNo difference in time points (same set of times).")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_time_extra.py fileA.nc fileB.nc")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
