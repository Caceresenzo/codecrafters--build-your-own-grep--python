import sys

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
        found |= handle_line(pattern, sys.stdin.read())

    else:
        for file in files:
            try:
                with open(file, "r") as fd:
                    for line in fd:
                        found |= handle_line(pattern, line.strip())
            except FileNotFoundError:
                print(f"grep: {file}: No such file or directory", file=sys.stderr)

    if not found:
        exit(1)


def handle_line(pattern: Pattern, line: str) -> bool:
    matcher = pattern.matcher(line)

    if not matcher.find(0):
        return False

    print(line)
    return True


if __name__ == "__main__":
    main()
