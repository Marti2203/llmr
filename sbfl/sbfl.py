import glob
import sys
import polars as pl
import os
import subprocess as sp
import numpy as np


if sys.argv[1] == "-h" or sys.argv[1] == "--help":
    print("""Usage:
        python sbfl <dir with crashing traces> <dir with non crashing traces>
    """)
    exit(0)

def traces_to_df(dirname_a, dirname_b, crashing_name):
    crashing_trace_fnames     = glob.glob(f"{dirname_a}/*")
    non_crashing_trace_fnames = glob.glob(f"{dirname_b}/*")

    crashing_traces = []
    for fname in crashing_trace_fnames:
        print(fname)
        sp.run(f'sed -i "s/|/\\n/g" {fname}', shell=True, executable="/bin/bash")
        df = pl.read_csv(fname, has_header=False, new_columns=["trace"]).group_by("trace").count()
        df = (df.select("count").transpose(column_names = df["trace"]))
        df = df.cast(pl.UInt32)
        df = df.select(sorted(df.columns))
        crashing_traces += [df]

    crashing_df = pl.concat(crashing_traces, how="diagonal").with_columns(
        pl.lit(True).alias(crashing_name)
    )

    non_crashing_traces = []
    for fname in non_crashing_trace_fnames:
        print(fname)
        sp.run(f'sed -i "s/|/\\n/g" {fname}', shell=True, executable="/bin/bash")
        df = pl.read_csv(fname, has_header=False, new_columns=["trace"]).group_by("trace").count()
        df = (df.select("count").transpose(column_names = df["trace"]))
        df = df.cast(pl.UInt32)
        df = df.select(sorted(df.columns))
        non_crashing_traces += [df]

    non_crashing_df = pl.concat(non_crashing_traces, how="diagonal").with_columns(
        pl.lit(False).alias(crashing_name)
    )

    df = pl.concat([non_crashing_df, crashing_df], how="diagonal")
    cols = df.columns
    cols.remove(crashing_name)
    df = df.select([crashing_name] + sorted(cols))
    df = df.fill_null(0)
    return df

crashing_name = os.path.basename(sys.argv[1].rstrip("/"))
out_dir = os.path.commonprefix([os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])])
df = traces_to_df(sys.argv[1], sys.argv[2], crashing_name)

df = df > 0

counts = df.group_by(crashing_name).count()

total_failing = counts.filter(pl.col(crashing_name))["count"][0]
total_passing = counts.filter(~pl.col(crashing_name))["count"][0]

print("")
print("total crashing traces:")
print(total_failing)
print("total non-crashing traces:")
print(total_passing)

efs = df.filter( pl.col(crashing_name)).drop(crashing_name).sum()
eps = df.filter(~pl.col(crashing_name)).drop(crashing_name).sum()
nfs = ((df.filter( pl.col(crashing_name)).drop(crashing_name)) == False).sum()
nps = ((df.filter(~pl.col(crashing_name)).drop(crashing_name)) == False).sum()

ochiai = efs / ((total_failing) * (efs + eps)).map_rows(np.sqrt)

scores = ochiai.transpose(include_header=True, header_name="line", column_names=["score"])
# scores.write_csv("all-ochiai.csv")
top_patches = int(os.getenv("TOP_PATCHES",20))
top5 =  scores.sort("score", descending=True).head(top_patches)

print("")
print("Ochiai Top-{}".format(top_patches))
print(top5)
top5.write_csv(f"{out_dir}/ochiai.csv", include_header=False)
