import sys

from app.regex import Pattern


def main():
    expression = sys.argv[2]
    input_line = sys.stdin.read()

    if sys.argv[1] != "-E":
        print("Expected first argument to be '-E'")
        exit(1)

    pattern = Pattern.compile(expression)
    matcher = pattern.matcher(input_line)

    if matcher.find(0):
        for number in range(matcher.group_count() + 1):
            print(f"group[{number}] = `{matcher.group(number)}`")

        exit(0)
    else:
        exit(1)


if __name__ == "__main__":
    main()
