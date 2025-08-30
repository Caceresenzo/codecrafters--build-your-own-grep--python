import sys
import typing
import os

import click

from app.regex import Pattern


@click.command()
@click.option("-E", "--extended-regexp", is_flag=True, help="PATTERNS are extended regular expressions")
@click.option("-r", "--recursive", is_flag=True, help="how to handle directories recursively")
@click.argument("pattern")
@click.argument("files", nargs=-1)
def main(
    extended_regexp: bool,
    recursive: bool,
    pattern: str,
    files: list[str],
):
    if not extended_regexp:
        print("grep: -E must be used")
        exit(1)

    pattern: Pattern = Pattern.compile(pattern)

    if recursive:
        files = list(traverse(files))

    found = False

    if not len(files):
        found |= handle_lines(pattern, sys.stdin)

    else:
        for file in files:
            file_name = None if len(files) == 1 else file

            try:
                with open(file, "r") as fd:
                    found |= handle_lines(
                        pattern,
                        fd,
                        file_name
                    )
            except FileNotFoundError:
                print(f"grep: {file}: No such file or directory", file=sys.stderr)

    if not found:
        exit(1)


def traverse(files: list[str]) -> typing.Generator[None, None, str]:
    for file in files:
        if not os.path.exists(file):
            yield file
        
        elif os.path.isdir(file):
            for root, _, filenames in os.walk(file):
                for filename in filenames:
                    yield os.path.normpath(os.path.join(root, filename))
        
        else:
            yield file


def handle_lines(
    pattern: Pattern,
    fd: typing.TextIO,
    file_name: str | None = None,
) -> bool:
    found = False

    for line in fd:
        line = line.rstrip("\n")
        matcher = pattern.matcher(line)

        if matcher.find(0):
            found = True

            if file_name:
                print(f"{file_name}:{line}")
            else:
                print(line)

    return found


if __name__ == "__main__":
    main()
