import argparse
import os
import rich
import sys
import os.path
import requests
import subprocess
import openai
from pprint import pprint

PROMPT = """
Here is the following file {lang_info} with name {file_name}.
There is a bug {fault_info}.
I would like you to repair the file and only respond with a patch in diff format with the file name to be the same as the original but in a different directory.
--- original/{file_name}
+++ patched/{file_name}
Here is the code:

{program}
"""


def send_prompt(text: str, key: str, num=1, temperature=0.8):
    # print(text)
    openai.api_key = key
    return openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful programming assistant, which repairs programs and outputs the changes in diff format, surrounded by triple backticks (```).",
            },
            {"role": "user", "content": text},
        ],
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
    # out is the output of the command, and err is the exit value
    return int(process.returncode)


def build(build_script):
    res = execute_command(build_script)
    return res == 0


def test(test_script):
    res = execute_command(test_script)
    return res == 0


def apply_patch(file, patch_path):
    res = execute_command("patch -p1 < {}".format(patch_path), directory=os.path.dirname(file))
    return res == 0


def parse_args():
    parser = argparse.ArgumentParser(prog="flem", usage="%(prog)s [options]")
    parser._action_groups.pop()
    required = parser.add_argument_group("Required arguments")
    required.add_argument(
        "-file",
        help="Location of faulty file",
        type=str,  # argparse.FileType("r"),
        required=True,
    )

    optional = parser.add_argument_group("Optional arguments")

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
        "-patches",
        help="Amount of generated patches",
        type=int,
        required=False,
        default=1,
    )

    optional.add_argument(
        "-test",
        help="Build script location",
        type=str,  # argparse.FileType("r"),
        required=False,
    )
    optional.add_argument(
        "-fl", help="Fault localization path", type=argparse.FileType("r")
    )
    optional.add_argument("-d", "-debug", help="Run in debug mode")
    optional.add_argument("-lang", help="Lanaguage")
    optional.add_argument(
        "-lines",
        help="Whether to write down the specific lines. This works only if there is fault localization given.",
        action="store_true",
        default=False,
    )
    # optional.add_argument("-tests", help="Tests")
    return parser.parse_args()


def repair(args):
    lang_info = "in the {} programming language".format(
        args.lang if args.lang else args.file.split(".")[-1]
    )
    bug_info = "in the file"
    if not os.path.exists(args.file):
        print("FILE DOES NOT EXIST")
        sys.exit(1)
    with open(args.file) as f:
        file_contents = f.read()
    if args.fl:
        fl_info = list(map(lambda line: line.split("@")[-1], args.fl.readlines()))
        if args.lines:
            fl_info = list(
                map(lambda line: "`" + file_contents[line - 1] + "`", fl_info)
            )
        bug_info = "in any of the lines in the list {}".format(",".join(fl_info))

    response = send_prompt(
        PROMPT.format(
            lang_info=lang_info,
            file_name=os.path.basename(args.file),
            program=file_contents,
            fault_info=bug_info,
        ),
        key="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",
        num=args.patches,
    )["choices"]

    os.makedirs(args.output, exist_ok=True)

    for i, resp in enumerate(response):
        patch_path = os.path.join(args.output, "patch_{}".format(i))
        patch = resp["message"]["content"].split("```")[1]
        if patch.startswith("diff\n"):
            patch = patch[5:]
        print(patch)
        with open(patch_path, "w") as f:
            f.write(patch)
        # build
        if not apply_patch(args.file, patch_path):
            print("FAILED TO PATCH")
        if args.build:
            if build(args.build):
                print("Patch {} Compiles".format(i))
        # test
        if args.test:
            if test(args.test):
                print("Patch {} is Plausible".format(i))


if __name__ == "__main__":
    repair(parse_args())
