import abc
import dataclasses
import enum
import sys
import typing


class CharacterClass(enum.Enum):

    DIGITS = "d"
    WORDS = "w"

    def test(self, character: str):
        character = ord(character)

        match self:
            case CharacterClass.DIGITS:
                return character >= ord('0') and character <= ord('9')

            case CharacterClass.WORDS:
                return (
                    (character >= ord('0') and character <= ord('9'))
                    or character >= ord('A') and character <= ord('Z')
                    or character >= ord('a') and character <= ord('z')
                    or character == ord('_')
                )

            case _:
                raise RuntimeError(f"unknown enum: {self}")


@dataclasses.dataclass
class Consumer:

    input: str
    index: int = 0
    marks: typing.List[int] = dataclasses.field(default_factory=list)

    def next(self):
        try:
            character = self.input[self.index]
            self.index += 1

            return character
        except IndexError:
            return "\0"

    def mark(self):
        self.marks.append(self.index)
        return self.index

    def reset(self):
        return self.marks.pop()


class Matcher(abc.ABC):

    @abc.abstractmethod
    def test(self, input: str):
        pass


@dataclasses.dataclass
class Literal(Matcher):

    value: str

    def test(self, input: Consumer):
        for character in self.value:
            next = input.next()
            print(f"testing next `{next}` with `{character}`")
            if next != character:
                return False

        return True


@dataclasses.dataclass
class Range(Matcher):

    character_class: CharacterClass

    def test(self, input: Consumer):
        return self.character_class.test(input.next())


@dataclasses.dataclass
class Node:

    name: str
    matchers: typing.List[typing.Tuple[Matcher, "Node"]] = dataclasses.field(default_factory=list)

    @property
    def end(self):
        return len(self.matchers) == 0

    def add(self, matcher: Matcher, node: "Node"):
        self.matchers.append((matcher, node))


def build(pattern):
    start = Node("start")
    end = Node("end")

    if len(pattern) == 1:
        literal = Literal(pattern[0])
        start.add(literal, end)
    elif pattern == '\\d':
        range = Range(CharacterClass.DIGITS)
        start.add(range, end)
    elif pattern == '\\w':
        range = Range(CharacterClass.WORDS)
        start.add(range, end)
    else:
        raise RuntimeError(f"Unhandled pattern: {pattern}")

    return start


def match(root: Node, input: Consumer) -> bool:
    if root.end:
        return True

    for matcher, node in root.matchers:
        input.mark()

        if matcher.test(input):
            return match(node, input)

        input.reset()

    return False


def main():
    pattern = sys.argv[2]
    input_line = sys.stdin.read()

    if sys.argv[1] != "-E":
        print("Expected first argument to be '-E'")
        exit(1)

    graph = build(pattern)
    print(graph)

    if match(graph, Consumer(input_line)):
        exit(0)
    else:
        exit(1)


if __name__ == "__main__":
    main()
