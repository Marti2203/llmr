import sys
import os
import glob
import subprocess as sp

if sys.argv[1] == "-h" or sys.argv[1] == "--help":
    print("""Usage:
        python get_all_traces.py <dir with crashing inputs> <dir with non crashing inputs> -- <afl style command to invoke input>

        The afl-style command should have @@ instead of the input file name.
        Traces will be output in the `<prog_name>_traces` directory.
    """)
    exit(0)

crashing_fnames     = (glob.glob(f"{sys.argv[1]}/*"))
non_crashing_fnames = (glob.glob(f"{sys.argv[2]}/*"))

if sys.argv[3] == "--":
    afl_cmd = " ".join(sys.argv[4:])
    out_dirname = f"{os.path.splitext(sys.argv[4])[0]}_traces"
else:
    afl_cmd = " ".join(sys.argv[3:])
    out_dirname = f"{os.path.splitext(sys.argv[3])[0]}_traces"

os.makedirs(out_dirname)

sub_dirname = os.path.basename(sys.argv[1].rstrip("/"))
os.makedirs(f"{out_dirname}/{sub_dirname}", exist_ok=True)
for fname in crashing_fnames:
    trace_fname = f"{out_dirname}/{sub_dirname}/{os.path.basename(fname)}.trace"
    print(trace_fname)
    sp.run(f'''TRACE_FILE={trace_fname} ./{afl_cmd.replace("@@", fname)}''', shell=True, executable="/bin/bash")
    print("----")

sub_dirname = os.path.basename(sys.argv[2].rstrip("/"))
os.makedirs(f"{out_dirname}/{sub_dirname}", exist_ok=True)
for fname in non_crashing_fnames:
    trace_fname = f"{out_dirname}/{sub_dirname}/{os.path.basename(fname)}.trace"
    print(trace_fname)
    sp.run(f'''TRACE_FILE={trace_fname} ./{afl_cmd.replace("@@", fname)}''', shell=True, executable="/bin/bash")
    print("----")

