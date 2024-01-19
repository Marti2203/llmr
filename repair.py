import argparse
import heapq
import json
import math
import os.path
import shutil
import subprocess
import sys
from os.path import join
from typing import cast
from typing import Dict
from typing import List
from typing import Literal
from typing import Tuple
from typing import Union

import rich
from openai import OpenAI

MAX_TOKENS = 4096
MAX_NON_CHUNKED_LENGTH = 4000
file_error_log = "error.logs"


def get_client(key: str):
    client = OpenAI(api_key=key)
    return client


from pprint import pprint

SINGLE_PROMPT = """
Here is the following file {lang_info} with name {file_name}.
There is a bug {fault_info}.
I would like you to repair the file and respond with a patched version of the file.
Here is the code:

{program}
"""

CHUNKED_PROMPT = """
Here is the following file {lang_info} with name {file_name}.
There is a bug {fault_info}.
I would like you to repair the file and respond with a patched version of the file.
Here is the code:

{program}
"""

CONTEXT_PROMPT = """
Here is a fragment from the following file {lang_info} with name {file_name}.
There is a functionality-related bug {fault_info}.
I would like you to repair the fragment and respond with a patched version of the fragment.
Focus on the current segment only and what needs to be changed in it to make it inserted back at the right spot, which will ensure it makes a syntactically correct file.
Here is the code:

{program}
"""

prompt_number = 0


def send_prompt_single(
    args,
    text: str,
    model: str,
    key="",
    num=1,
    temperature=0.8,
    reference=None,
    bug_description=None,
    lines=None,
):
    lines = lines or []
    global prompt_number

    initial_message = "You are a helpful programming assistant, which repairs programs and outputs the contents of the repaired file, surrounded by triple backticks (```)."
    text_fragments = [{"role": "user", "content": text}]
    messages = [
        {
            "role": "system",
            "content": initial_message,
        },
        *text_fragments,
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
                "content": f"This is the description of the task the program has to handle, your goal is to satisfy it:\n {bug_description}",
            },
        )

    prompt_number += 1
    if args.debug:
        with open(join(args.output, "prompt_{}.json".format(prompt_number)), "w") as f:
            json.dump(messages, f, indent=4)

    return get_client(key).chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=MAX_TOKENS,
        top_p=1.0,
        n=num,
    )


def send_prompt_chunked(
    args,
    text: str,
    model: str,
    num=1,
    key="",
    temperature=0.8,
    reference=None,
    bug_description=None,
    lines=None,
):
    lines = lines or []
    global prompt_number

    print("Chunking. Currently not reachable!!!!!!")
    exit(1)
    initial_message = 'You are a helpful programming assistant, which repairs programs provied in chunks and outputs the contents of the repaired chunks, surrounded by triple backticks (```). with the first line having saying which is the chunk in the format "CHUNK x" relating it to the originally provided chunk'
    chunks = math.ceil(len(text) / MAX_NON_CHUNKED_LENGTH)
    text_fragments = []

    for chunk in range(chunks):
        chunk_text = text[
            chunk
            * MAX_NON_CHUNKED_LENGTH : min(
                len(text), (chunk + 1) * MAX_NON_CHUNKED_LENGTH
            )
        ]
        text_fragments.append(
            {"role": "user", "content": f"Chunk {chunk+1}/{chunks}:\n{chunk_text}"}
        )

    messages = [
        {
            "role": "system",
            "content": initial_message,
        },
        *text_fragments,
    ]

    if reference:
        chunks = math.ceil(len(reference) / MAX_NON_CHUNKED_LENGTH)
        for chunk in reversed(range(chunks)):
            chunk_text = reference[
                chunk
                * MAX_NON_CHUNKED_LENGTH : min(
                    len(reference), (chunk + 1) * MAX_NON_CHUNKED_LENGTH
                )
            ]
            messages.insert(
                1,
                {
                    {
                        "role": "system",
                        "content": f"Reference Chunk {chunk+1}/{chunks}:\n{chunk_text}",
                    }
                },
            )

        messages.insert(
            1,
            {
                "role": "system",
                "content": "This is the reference file which you can use as a solution to help you repair. It is also provided in chunks. The chunks need not align with the original file.",
            },
        )

    if bug_description:
        messages.insert(
            1,
            {
                "role": "system",
                "content": f"This is the description of the task the program has to handle, your goal is to satisfy it:\n {bug_description}",
            },
        )

    prompt_number += 1
    if args.debug:
        with open(join(args.output, "prompt_{}.json".format(prompt_number)), "w") as f:
            json.dump(messages, f, indent=4)

    return get_client(key).chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=MAX_TOKENS,
        top_p=1.0,
        n=num,
    )


def send_prompt_context(
    args,
    text: str,
    model: str,
    num=1,
    key="",
    temperature=0.8,
    reference=None,
    bug_description=None,
    lines=None,
):
    lines = lines or []
    global prompt_number

    initial_message = (
        "You are a helpful programming assistant, which repairs fragments of programs and outputs the contents of the repaired fragment, surrounded by triple backticks (```). "
        "You keep the indentation the same as original to allow for the fragment to be succesfully inserted back into the file."
        "Assume that after the fragment there is code that will make it syntactically correct."
    )
    text_fragments = [{"role": "user", "content": text}]

    messages = [
        {
            "role": "system",
            "content": initial_message,
        },
        *text_fragments,
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
                "content": f"This is the description of the task the program has to handle, your goal is to satisfy it:\n {bug_description}",
            },
        )

    prompt_number += 1
    if args.debug:
        with open(join(args.output, "prompt_{}.json".format(prompt_number)), "w") as f:
            json.dump(messages, f, indent=4)

    return get_client(key).chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=MAX_TOKENS,
        top_p=1.0,
        n=num,
    )


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
    parser = argparse.ArgumentParser(prog="LLMR", usage="%(prog)s [options]")
    parser._action_groups.pop()

    file_processing_approach = parser.add_mutually_exclusive_group(required=False)

    file_processing_approach.add_argument(
        "-single",
        help="Process the file as a single unit. Will fail if too big to process by the model",
        action="store_true",
        default=True,
    )

    file_processing_approach.add_argument(
        "-chunked",
        help="Split the file in chunks when too big",
        action="store_true",
        default=False,
    )

    file_processing_approach.add_argument(
        "-context",
        help="Send the faulty lines using a context window. Truncates the code to the specified lines",
        type=int,
        default=-1,
    )

    fault_localization_group = parser.add_mutually_exclusive_group(required=True)

    fault_localization_group.add_argument(
        "-file",
        help="Location of faulty file, file level granularity. Does not work with context window",
        type=str,
        required=False,
    )

    fault_localization_group.add_argument(
        "-do-fl",
        help="Whether to do fault localization internally",
        action="store_true",
        default=False,
    )

    fault_localization_group.add_argument(
        "-fl-data", help="Path of fautl localization data", type=argparse.FileType("r")
    )

    optional = parser.add_argument_group("Arguments")
    optional.add_argument(
        "-key",
        help="OpenAI key. Preferably use OPEN_AI_KEY environment variable",
        type=str,
        required=False,
        default="",
    )

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
        "-only-plausible",
        help="Only plausible patches are generated",
        action="store_true",
        required=False,
        default=False,
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
        help="Path to a description of the bug. Can help the model create a better patch.",
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
        type=str,
        required=False,
    )
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
        "-binary-loc",
        help="Location of the binary",
        type=str,
        required=False,
        default=None,
    )

    optional.add_argument(
        "-fl-formula", help="Fault localization formula", type=str, default="Ochiai"
    )

    optional.add_argument(
        "--project-path", help="Project path", type=str, required=False, default=None
    )

    optional.add_argument(
        "-d", "--debug", help="Run in debug mode", action="store_true", default=False
    )
    optional.add_argument("-lang", "--language", help="Lanaguage")
    optional.add_argument(
        "-lines",
        help="Whether to write down the specific lines when giving information to the model. This works only if there is fault localization with line level granularity.",
        action="store_true",
        default=False,
    )

    return parser.parse_args()


def repair(
    args, file_path: str, faulty_lines: List[int]
) -> List[Tuple[Literal["failed", "plausible", "compiling", "noncompiling"], str, str]]:
    lang_info = "in the {} programming language".format(
        args.language if args.language else os.path.splitext(file_path)[1]
    )

    if not os.path.exists(file_path):
        print(
            "File {} does not exist, yet it is passed as the target for repair".format(
                file_path
            )
        )
        return []

    with open(file_path) as f:
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

    if args.chunked and len(file_path) > MAX_NON_CHUNKED_LENGTH:
        print("File is too large to be processed. Chunking to be implemented soon")
        return []

    results = []
    response_count = 0
    for iteration in range(args.iterations):
        print("\n\nIteration {} on file {}\n\n".format(iteration, file_path))

        if args.chunked:
            instances, producer, consumer, prompt, extra_args = (
                [],
                send_prompt_chunked,
                process_response_chunked,
                CHUNKED_PROMPT,
                {},
            )

        elif args.context > 0:
            instances_temp = []
            for faulty_line in faulty_lines:
                temp_info = "in the line {}".format(file_contents[faulty_line - 1])
                file_lines = file_contents.split("\n")
                temp_program = "\n".join(
                    file_lines[
                        max(0, faulty_line - args.context) : min(
                            len(file_lines) - 1, faulty_line + args.context
                        )
                    ]
                )
                with open(
                    join(
                        args.output,
                        "context_{}_{}".format(
                            faulty_line, os.path.basename(file_path)
                        ),
                    ),
                    "w",
                ) as f:
                    f.write(temp_program)

                instances_temp.append((temp_info, temp_program))

            instances, producer, consumer, prompt, extra_args = (
                instances_temp,
                send_prompt_context,
                process_response_context,
                CONTEXT_PROMPT,
                {"lines": faulty_lines},
            )
        elif args.single:
            temp_info = ""
            if args.file:
                temp_info = "in the file"
            elif args.lines:
                temp_info = "in any of the lines in the list [{}]".format(
                    ",".join(
                        list(map(lambda x: file_contents[int(x) - 1], faulty_lines))
                    )
                )
            else:
                temp_info = "in any of the lines in the list [{}]".format(
                    ",".join(map(str, faulty_lines))
                )

            instances, producer, consumer, prompt, extra_args = (
                [(temp_info, file_contents)],
                send_prompt_single,
                process_response_single,
                SINGLE_PROMPT,
                {},
            )
        else:
            print("Unexpected file processing case, LLMR needs to be examined")
            exit(1)

        for instance, (fault_info, program) in enumerate(instances):
            response = producer(
                args,
                text=prompt.format(
                    lang_info=lang_info,
                    file_name=os.path.basename(file_path),
                    program=program,
                    fault_info=fault_info,
                ),
                key=args.key or os.environ.get("OPEN_AI_KEY", ""),
                num=args.patches,
                model=args.model,
                reference=reference_contents,
                bug_description=description_contents,
            ).choices
            for resp in response:
                print("\n\nProcessing response {}\n\n".format(response_count))
                try:
                    results.append(
                        consumer(
                            resp,
                            file_path,
                            args,
                            response_count,
                            instance=instance,
                            **extra_args,
                        )
                    )
                except Exception as e:
                    print("Got exception while processing response")
                    print(e)
                finally:
                    with open(file_path, "w") as f:
                        f.write(file_contents)
                response_count = response_count + 1
    return results


def make_patch(args, file: str, patched_path: str, patch_index: int):
    os.system(
        f"diff -u {file} {patched_path} > {args.output}/patches/patch_{patch_index}.diff"
    )


conversion_table = {
    "failed": 1,
    "noncompiling": 0,
    "implausible": -1,
    "plausible": -2,
}


def process_response_single(resp, file: str, args, response_count: int, **kwargs):
    with open(
        os.path.join(
            args.output,
            "response_{}_{}.txt".format(os.path.basename(file), response_count),
        ),
        "w",
    ) as f:
        f.write(resp.message.content)
    # if args.debug:
    #    print(resp.message.content)
    patched_path = os.path.join(
        args.output, "patched_{}_{}".format(response_count, os.path.basename(file))
    )
    if "```" not in resp.message.content:
        print("Skipping output {} due to missing concrete patch".format(response_count))
        return ("failed", file, None)
    patched_file: str = resp.message.content.split("```")[1]

    first_line = patched_file[: patched_file.index("\n") + 1].lower()
    if args.language and first_line.startswith(args.language):
        patched_file = patched_file[len(args.language) :]
    if patched_file.startswith("\n"):
        patched_file = patched_file[1:]
    with open(patched_path, "w") as f:
        f.write(patched_file)
    return evaluate_patched_file(args, patched_path, patched_file, response_count, file)


def process_response_context(resp, file: str, args, response_count: int, **kwargs):
    with open(
        os.path.join(
            args.output,
            "response_{}_{}.txt".format(os.path.basename(file), response_count),
        ),
        "w",
    ) as f:
        f.write(resp.message.content)

    patched_context_path = os.path.join(
        args.output,
        "patched_context_{}_{}".format(response_count, os.path.basename(file)),
    )

    patched_path = os.path.join(
        args.output, "patched_{}_{}".format(response_count, os.path.basename(file))
    )

    if "```" not in resp.message.content:
        print("Skipping output {} due to missing concrete patch".format(response_count))
        return ("failed", file, None)
    patched_context: str = resp.message.content.split("```")[1]

    first_line = patched_context[: patched_context.index("\n") + 1].lower()
    if args.language and first_line.startswith(args.language):
        patched_context = patched_context[len(args.language) :]
    if patched_context.startswith("\n"):
        patched_context = patched_context[1:]

    with open(patched_context_path, "w") as f:
        f.write(patched_context)

    with open(file, "r") as f:
        lines = f.readlines()
        context_size = args.context

        faulty_line = kwargs["lines"][kwargs["instance"]]

        bad_range = set(
            range(
                max(0, faulty_line - context_size),
                min(len(lines), faulty_line + context_size),
            )
        )

        for i in range(len(lines)):
            if i in bad_range:
                lines[i] = ""
            else:
                lines[i] = lines[i].rstrip()

        lines[max(0, faulty_line - context_size)] = patched_context + "\n"

        patched_file = "\n".join(lines)

    with open(patched_path, "w") as f:
        f.write(patched_file)

    return evaluate_patched_file(args, patched_path, patched_file, response_count, file)


def process_response_chunked(resp, file: str, args, response_count: int, **kwargs):
    with open(
        os.path.join(
            args.output,
            "response_{}_{}.txt".format(os.path.basename(file), response_count),
        ),
        "w",
    ) as f:
        f.write(resp.message.content)
    # if args.debug:
    #    print(resp.message.content)
    patched_path = os.path.join(
        args.output, "patched_{}_{}".format(response_count, os.path.basename(file))
    )
    if "```" not in resp.message.content:
        print("Skipping output {} due to missing concrete patch".format(response_count))
        return ("failed", file, None)
    patched_file: str = resp.message.content.split("```")[1]

    first_line = patched_file[: patched_file.index("\n") + 1].lower()
    if args.language and first_line.startswith(args.language):
        patched_file = patched_file[len(args.language) :]
    if patched_file.startswith("\n"):
        patched_file = patched_file[1:]
    with open(patched_path, "w") as f:
        f.write(patched_file)
    return evaluate_patched_file(args, patched_path, patched_file, response_count, file)


def evaluate_patched_file(args, patched_path, patched_file, i, file):
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
                return process_fault_localization(distribution)
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
        execute_command(
            f"bash -c 'for x in $(find {args.project_path} | grep -E \".*\\.(c|cxx|hxx|cpp|hpp|h)$\"); do python3 /sbfl/dump_lines.py $x $(cat $x | wc -l ) >> /sbfl/lines.txt ; done'",
            directory="/sbfl/",
        )

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

    with open(join(args.output, "ochiai.csv"), "r") as f:
        distribution = {}
        for line in f.readlines():
            path, line_and_prob = line.split(":")
            line, _ = line_and_prob.split(",")
            distribution[path] = distribution.get(path, []) + [line]
        return process_fault_localization(distribution)


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
        return process_fault_localization(distribution)


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


def process_fault_localization(
    fault_localization_distribution: Dict[str, List[Union[str, int]]]
) -> List[Tuple[str, List[int]]]:
    distribution_converted = []
    for k, v in fault_localization_distribution.items():
        distribution_converted.append((k, list(map(int, v))))
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

        return process_fault_localization(distribution)
    elif args.file:
        if args.context > 0:
            print("Context window is non-zero but only info provided is file")
            exit(1)
        return cast(List[Tuple[str, List[int]]], [(args.file, [])])
    else:
        print(
            "Please provide a way to find the file to repair - either fault localization (manual, provided) or file path"
        )
        sys.exit(1)


if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        print(args)
    os.makedirs(os.path.join(args.output, "patches"), exist_ok=True)

    if args.file and not os.path.exists(args.file):
        print(
            "File {} does not exist, yet it is passed as the target".format(args.file)
        )
        sys.exit(1)

    cases = get_bug_info(args)

    patches: List[Tuple[int, str, str]] = []

    for file_path, faulty_lines in cases:
        for patch in repair(args, file_path, faulty_lines):
            transformed_patch = (conversion_table[patch[0]], *patch[1:])
            heapq.heappush(patches, transformed_patch)

    patch_index = 0
    for _ in range(min(args.top, len(patches))):
        (priority, file, patched_path) = heapq.heappop(patches)
        if patched_path is None:
            break
        if not args.only_plausible or priority == conversion_table["plausible"]:
            make_patch(args, file, patched_path, patch_index)
            patch_index = patch_index + 1
