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
I would like you to repair the file and respond with a patched version of the file.
Here is the code:

{program}
"""


def send_prompt(text: str, key: str, model: str, num=1, temperature=0.8):
    # print(text)
    openai.api_key = key
    return openai.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful programming assistant, which repairs programs and outputs the contents of the repaired file, surrounded by triple backticks (```).",
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
    if show_output:
        print(output)
    # out is the output of the command, and err is the exit value
    return int(process.returncode)


def build(build_script,debug=False):
    res = execute_command("./" + build_script,show_output=debug)
    return res == 0


def test(test_script, args=None,debug=False):
    res = execute_command("./" + test_script + ' ' + args if args is not None else ' ', show_output=debug)
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
        "-model",
        help="OpenAI model to be used",
        type=str,
        required=False,
        default="gpt-4"
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
        "-patches",
        help="Amount of generated patches",
        type=int,
        required=False,
        default=1,
    )

    optional.add_argument(
        "-test",
        help="Test script location",
        type=str,  # argparse.FileType("r"),
        required=False,
    )

    optional.add_argument('-tests',help="Tests", type=str,required=False)

    optional.add_argument(
        "-fl", help="Fault localization path", type=argparse.FileType("r")
    )
    optional.add_argument("-d", "--debug", help="Run in debug mode", action="store_true", default=False)
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
        model=args.model,
    )["choices"]

    os.makedirs(args.output, exist_ok=True)

    for i, resp in enumerate(response):
        print("\n\nProcessing response {}\n\n".format(i))
        if args.debug:
            with open(os.path.join(args.output, "response_{}.txt".format(i)),"w") as f:
                f.write(resp["message"]["content"])
            print(resp["message"]["content"])
        patched_path = os.path.join(args.output, "patched_{}_{}".format(i,os.path.basename(args.file)))
        if '```' not in resp["message"]["content"]:
            print("Skipping output {}".format(i))
            continue
        patched_file = resp["message"]["content"].split("```")[1]
        if args.lang and patched_file.startswith(args.lang):
            patched_file = patched_file[len(args.lang):]
        if patched_file.startswith('\n'):
            patched_file = patched_file[1:]
        print(patched_file)
        with open(patched_path, "w") as f:
            f.write(patched_file)
        # build
        #if not apply_patch(args.file, patch_path):
        #    print("FAILED TO PATCH")
        try:
            with open(args.file,'w') as f:
                f.write(patched_file)
            built = False
            if args.build:
                if build(args.build,args.debug):
                    built=True
                    print("Patch {} Compiles".format(i))
                else:
                    print("Patch {} does not compile".format(i))
                    shutil.move(patched_path,patched_path+'_noncompiling')
            else:
                built = True
            if built and args.test:
                plausible = True
                if args.tests:
                    for test_id in args.tests.split(','):
                        if not test(args.test,test_id,args.debug):
                            print("Patch {} failed for test {}".format(i,test_id))
                            shutil.move(patched_path,patched_path+'_implausible')
                            plausible=False
                            break
                else:
                    if not test(args.test,None,args.debug):
                        print("Patch {} failed test script".format(i))
                        plausible = False
                        shutil.move(patched_path,patched_path+'_implausible')
                if plausible:
                    print("Patch {} is Plausible".format(i))
                    shutil.move(patched_path,patched_path+'_plausible')
        finally:
            with open(args.file,'w') as f:
                f.write(file_contents)
if __name__ == "__main__":
    repair(parse_args())
