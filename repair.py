import argparse
import os
from typing import Dict, List
import rich
import sys
import os.path
import shutil
import requests
import subprocess
from openai import OpenAI
from os.path import join
import json
import heapq

client = OpenAI(api_key="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")
from pprint import pprint

PROMPT = """
Here is the following file {lang_info} with name {file_name}.
There is a bug {fault_info}.
I would like you to repair the file and respond with a patched version of the file.
Here is the code:

{program}
"""

prompt_number = 0


def send_prompt(
    args,
    text: str,
    model: str,
    num=1,
    temperature=0.8,
    reference=None,
    bug_description=None,
):
    global prompt_number
    # print(text)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful programming assistant, which repairs programs and outputs the contents of the repaired file, surrounded by triple backticks (```).",
        },
        {"role": "user", "content": text},
    ]
    if reference:
        messages.insert(
            1,
            {
                "role": "system",
                "content": f"This is the reference file, which you can use as a solution to help you repair:\n {reference}",
            },
        )
    if bug_description:
        messages.insert(
            1,
            {
                "role": "system",
                "content": f"This is the description of the bug, your goal is to satisfy it:\n {bug_description}",
            },
        )

    prompt_number += 1
    if args.debug:
        with open(join(args.output, "prompt_{}.json".format(prompt_number)), "w") as f:
            json.dump(messages, f, indent=4)

    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=2048,
        top_p=1.0,
        n=num,
    )


file_error_log = "error.logs"


def execute_command(command: str, show_output=True, env=dict(), directory=None):
    # Print executed command and execute it in console
    command = command.encode().decode("ascii", "ignore")
    if not directory:
        directory = os.getcwd()
    print_command = "[{}] {}".format(directory, command)
    print("Executing ", print_command)
    command = "{{ {} ;}} 2> {}".format(command, file_error_log)
    if not show_output:
        command += " > /dev/null"
    # print(command)
    new_env = os.environ.copy()
    new_env.update(env)
    process = subprocess.Popen(
        [command], stdout=subprocess.PIPE, shell=True, env=new_env, cwd=directory
    )
    (output, error) = process.communicate()
    if show_output:
        print(output)
    # out is the output of the command, and err is the exit value
    return int(process.returncode)


def build(build_script, debug=False):
    res = execute_command(build_script, show_output=debug)
    return res == 0


def test(test_script, args=None, debug=False, env=dict()):
    res = execute_command(
        test_script + " " + args if args is not None else " ",
        show_output=debug,
        env=env,
    )
    return res == 0


def apply_patch(file, patch_path):
    res = execute_command(
        "patch -p1 < {}".format(patch_path), directory=os.path.dirname(file)
    )
    return res == 0


def parse_args():
    parser = argparse.ArgumentParser(prog="flem", usage="%(prog)s [options]")
    parser._action_groups.pop()

    optional = parser.add_argument_group("Arguments")

    optional.add_argument(
        "-iterations",
        help="Amount of iterations for a self-consistency check",
        type=int,
        required=False,
        default=1,
    )

    optional.add_argument(
        "-model",
        help="OpenAI model to be used",
        type=str,
        required=False,
        default="gpt-4-32k",
    )

    optional.add_argument(
        "-build",
        help="Build script location",
        type=str,  # argparse.FileType("r"),
        required=False,
    )

    optional.add_argument(
        "-output",
        "-o",
        help="Output directory",
        type=str,
        required=False,
        default="output",
    )

    optional.add_argument(
        "-description",
        help="Description of the bug",
        type=str,
        required=False,
        default=None,
    )

    optional.add_argument(
        "-patches",
        help="Amount of generated patches by model",
        type=int,
        required=False,
        default=1,
    )

    optional.add_argument(
        "-top",
        help="Top k patches to be selected",
        type=int,
        required=False,
        default=5,
    )

    optional.add_argument(
        "-test",
        help="Test script location",
        type=str,  # argparse.FileType("r"),
        required=False,
    )

    # optional.add_argument(
    #    "-context", help="Context Window size", type=int, required=False, default=-1
    # )

    optional.add_argument(
        "-reference", help="Reference file", type=str, required=False, default=None
    )

    optional.add_argument(
        "-passing-tests",
        help="Comma sepearted list of test identifiers for passing tests",
        type=str,
        required=False,
        default="",
    )
    optional.add_argument(
        "-failing-tests",
        help="Comma sepearted list of test identifiers for failing tests",
        type=str,
        required=False,
        default="",
    )

    optional.add_argument(
        "-fl-data", help="Fault localization path", type=argparse.FileType("r")
    )

    optional.add_argument(
        "-binary-loc",
        help="Location of the binary",
        type=str,
        required=False,
        default=None,
    )

    optional.add_argument(
        "-do-fl",
        help="Whether to do fault localization",
        action="store_true",
        default=False,
    )

    optional.add_argument(
        "-fl-formula", help="Fault localization formula", type=str, default="Ochiai"
    )

    optional.add_argument(
        "--project-path", help="Project path", type=str, required=False, default=None
    )

    optional.add_argument(
        "-file",
        help="Location of faulty file",
        type=str,
        required=False,
    )

    optional.add_argument(
        "-d", "--debug", help="Run in debug mode", action="store_true", default=False
    )
    optional.add_argument("-lang", "--language", help="Lanaguage")
    optional.add_argument(
        "-lines",
        help="Whether to write down the specific lines. This works only if there is fault localization given.",
        action="store_true",
        default=False,
    )
    # optional.add_argument("-tests", help="Tests")
    return parser.parse_args()


def repair(args, file:str, fault_info:str):
    lang_info = "in the {} programming language".format(
        args.language if args.language else file(".")[-1]
    )
    with open(file) as f:
        file_contents = f.read()

    os.makedirs(args.output, exist_ok=True)

    reference_contents = None
    description_contents = None

    if args.reference:
        with open(args.reference) as f:
            reference_contents = f.read()

    if args.description:
        with open(args.description) as f:
            description_contents = f.read()
    results = []
    i = 0
    for iteration in range(args.iterations):
        print("\n\nIteration {}\n\n".format(iteration))
        response = send_prompt(
            args,
            PROMPT.format(
                lang_info=lang_info,
                file_name=os.path.basename(file),
                program=file_contents,
                fault_info=fault_info,
            ),
            num=args.patches,
            model=args.model,
            reference=reference_contents,
            bug_description=description_contents,
        ).choices
        for resp in response:
            print("\n\nProcessing response {}\n\n".format(i))
            try:
                results.append(process_response(resp, file, args, i))
            finally:
                with open(file, "w") as f:
                    f.write(file_contents)
            i = i + 1
    return results


def make_patch(args, file:str, patched_path:str, i:int):
    os.system(f"diff -u {file} {patched_path} > {args.output}/patches/patch_{i}.diff")


conversion_table = {
    "failed": 1,
    "noncompiling": 0,
    "implausible": -1,
    "plausible": -2,
}


def process_response(resp, file:str, args, i:int):
    with open(os.path.join(args.output, "response_{}.txt".format(i)), "w") as f:
        f.write(resp.message.content)
    # if args.debug:
    #    print(resp.message.content)
    patched_path = os.path.join(
        args.output, "patched_{}_{}".format(i, os.path.basename(file))
    )
    if "```" not in resp.message.content:
        print("Skipping output {} due to missing concrete patch".format(i))
        return
    patched_file = resp.message.content.split("```")[1]
    if args.language and patched_file.startswith(args.language):
        patched_file = patched_file[len(args.language) :]
    if patched_file.startswith("\n"):
        patched_file = patched_file[1:]
    # print(patched_file)
    with open(patched_path, "w") as f:
        f.write(patched_file)
    os.makedirs(os.path.join(args.output, "patches"), exist_ok=True)
    # build
    # if not apply_patch(file, patch_path):
    #    print("FAILED TO PATCH")
    with open(file, "w") as f:
        f.write(patched_file)
    built = False
    if args.build:
        if build(args.build, args.debug):
            built = True
            print("Patch {} Compiles".format(i))
        else:
            print("Patch {} does not compile".format(i))
            shutil.copy(patched_path, patched_path + "_noncompiling")
            return ("noncompiling", file, patched_path)
    else:
        built = True

    if built and args.test:
        plausible = True
        if args.passing_tests or args.failing_tests:
            for test_id in [
                *args.passing_tests.split(","),
                *args.failing_tests.split(","),
            ]:
                if test_id:
                    if not test(args.test, test_id, args.debug):
                        print("Patch {} failed for test {}".format(i, test_id))
                        shutil.copy(patched_path, patched_path + "_implausible")
                        plausible = False
                        return ("implausible", file, patched_path)
        else:
            if not test(args.test, None, args.debug):
                print("Patch {} failed test script".format(i))
                plausible = False
                shutil.copy(patched_path, patched_path + "_implausible")
                return ("implausible", file, patched_path)
        if plausible:
            print("Patch {} is Plausible".format(i))
            shutil.copy(patched_path, patched_path + "_plausible")
            return ("plausible", file, patched_path)
    return ("failed", file, patched_path)


def do_fault_localization_py(args):
    execute_command(
        "python3 -m pytest --src . --family sbfl --exclude \"[$(ls | grep test | grep .py | tr '\n' ',' | sed 's/,$//')]\"",
        directory=args.project_path,
    )
    report_dir = os.path.dirname(args.project_path)
    for dir in os.listdir(report_dir):
        if dir.startswith("FauxPyReport"):
            if not os.path.exists(
                join(report_dir, dir, "Scores_{}.csv".format(args.fl_formula))
            ):
                print(
                    "Fault localization report for formula {} does not exist".format(
                        args.fl_formula
                    )
                )
                exit(1)
            with open(
                join(report_dir, dir, "Scores_{}.csv".format(args.fl_formula))
            ) as f:
                distribution = {}
                for line in f.readlines()[1:]:
                    # print(line)
                    path, line_and_probability = line.split("::")
                    line, _ = line_and_probability.split(",")
                    distribution[path] = distribution.get(path, []) + [line]
                return process_distribution(args, distribution)
    raise Exception("No fault localization report found")


def do_fault_localization_c(args):
    if not args.binary_loc:
        print(
            "Please provide the location of the binary for the binary if you want to do FL on C"
        )
        exit(1)

    os.makedirs(join(args.output, "passing_traces"), exist_ok=True)
    os.makedirs(join(args.output, "failing_traces"), exist_ok=True)

    
    if args.file:
        execute_command(
            "bash -c 'python3 /sbfl/dump_lines.py {0} $(cat {0} | wc -l) > /sbfl/lines.txt'".format(
                join(args.project_path, args.file)
            )
        )
    else:
        # Take all files and send them to the fault localization
        execute_command(f"bash -c 'for x in $(find {args.project_path} | grep -E \".*\\.(c|cxx|hxx|cpp|hpp|h)$\"); do python3 /sbfl/dump_lines.py $x $(cat $x | wc -l ) >> /sbfl/lines.txt ; done'",
        directory="/sbfl/")

    execute_command(
        "python3 /sbfl/instrument.py {} /sbfl/lines.txt".format(
            join(args.project_path, args.binary_loc)
        ),
        directory="/sbfl/",
    )

    execute_command(
        "bash -c 'mv /sbfl/*.tracer {}'".format(
            join(args.project_path, args.binary_loc)
        )
    )

    for passing_test_id in args.passing_tests.split(","):
        if passing_test_id:
            test(
                args.test,
                passing_test_id,
                args.debug,
                env={
                    "TRACE_FILE": join(
                        args.output,
                        "passing_traces",
                        f"passing_trace_{passing_test_id}",
                    )
                },
            )
    for failing_test_id in args.failing_tests.split(","):
        if failing_test_id:
            test(
                args.test,
                failing_test_id,
                args.debug,
                env={
                    "TRACE_FILE": join(
                        args.output,
                        "failing_traces",
                        f"failing_trace_{failing_test_id}",
                    )
                },
            )

    if (
        os.listdir(join(args.output, "passing_traces")) == 0
        and os.listdir(join(args.output, "failing_traces")) == 0
    ):
        print("No traces were generated")
        exit(1)

    execute_command(
        f"python3 /sbfl/sbfl.py {join(args.output,'failing_traces')} {join(args.output,'passing_traces')}",
        directory="/sbfl/",
    )

    with open("/output/ochiai.csv", "r") as f:
        distribution = {}
        for line in f.readlines():
            path, line_and_prob = line.split(":")
            line, prob = line_and_prob.split(",")
            distribution[path] = distribution.get(path, []) + [line]
        return process_distribution(args, distribution)


def do_fault_localization_java(args):
    execute_command(
        "java -cp '/flacoco/target/flacoco-1.0.7-SNAPSHOT-jar-with-dependencies.jar' fr.spoonlabs.flacoco.cli.FlacocoMain --projectpath {} -o flacoco.run".format(
            args.project_path
        ),
        directory=args.project_path,
    )
    with open(join(args.project_path, "flacoco.run")) as f:
        distribution = {}
        for line in f.readlines():
            path, line, _ = line.split(",")
            path = join(
                args.project_path,
                "src",
                "main",
                "java",
                path.replace(".", "/") + ".java",
            )
            distribution[path] = distribution.get(path, []) + [line]
        return process_distribution(args, distribution)


def fault_localization(args):
    if args.language.startswith("py"):
        return do_fault_localization_py(args)

    elif args.language.startswith("c"):
        return do_fault_localization_c(args)

    elif args.language.startswith("java"):
        return do_fault_localization_java(args)
    else:
        print("Unsupported language {}".format(args.language))
        exit(1)


def process_distribution(args, distribution: Dict[str, List[str]]):
    distribution_converted = []
    for k, v in distribution.items():
        if args.lines:
            with open(k) as f:
                lines = f.readlines()
                distribution_converted.append(
                    (
                        k,
                        "in any of the lines in the list [{}]".format(
                            ",".join(list(map(lambda x: lines[int(x) - 1], v)))
                        ),
                    )
                )
        else:
            distribution_converted.append(
                (k, "in any of the lines in the list {}".format(",".join(v)))
            )
    return distribution_converted


def get_bug_info(args):
    if args.do_fl:
        print("doing FL")
        required_args = [
            ("language", args.language),
            ("project path", args.project_path),
            ("test_script", args.test),
        ]
        for _, val in required_args:
            if not val:
                print(
                    f"Please provide {list(map(lambda x: x[0],required_args))} if you want fault localization"
                )
                exit(1)
        return fault_localization(args)
    elif args.fl_data:
        print("Reading FL data")
        distribution = {}
        for line in args.fl_data:
            path, line_and_probability = line.split("::")
            line, _ = line_and_probability.split(",")
            distribution[path] = distribution.get(path, []) + [line]
        
        return process_distribution(args, distribution)
    elif args.file:
        return [(args.file, "in the file ")]
    else:
        print(
            "Please provide a way to find the file to repair - either fault localization (manual, provided) or file path"
        )
        sys.exit(1)


if __name__ == "__main__":
    args = parse_args()
    print(args)

    if args.file and not os.path.exists(args.file):
        print("FILE {} DOES NOT EXIST".format(args.file))
        sys.exit(1)

    cases = get_bug_info(args)

    patches = []

    for file_path, fault_info in cases:
        for patch in repair(args, file_path, fault_info):
            transformed_patch = (conversion_table[patch[0]], *patch[1:])
            heapq.heappush(patches, patch)

    for i in range(min(args.top, len(patches))):
        (priority, file, patched_path) = heapq.heappop(patches)
        make_patch(args, file, patched_path, i)
