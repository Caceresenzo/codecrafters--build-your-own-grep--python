import sys
import typing

import click

from app.regex import Pattern


@click.command()
@click.option("-E", "--extended-regexp", is_flag=True, help="Use extended regular expressions")
@click.argument("expression")
@click.argument("files", nargs=-1)
def main(
    extended_regexp: bool,
    expression: str,
    files: list[str],
):
    if not extended_regexp:
        print("grep: -E must be used")
        exit(1)

    expression = sys.argv[2]
    pattern = Pattern.compile(expression)

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
