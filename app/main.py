import abc
import dataclasses
import enum
import pprint
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

    @staticmethod
    def of(character: str):
        for klass in CharacterClass:
            if klass.value == character:
                return klass

        raise RuntimeError(f"unknown CharacterClass: {character}")


@dataclasses.dataclass
class Consumer:

    input: str
    index: int = 0
    marks: typing.List[int] = dataclasses.field(default_factory=list)

    def next(self):
        if self.end:
            return "\0"

        previous_index = self.index
        self.index += 1

        return self.input[previous_index]

    def current(self):
        if self.end:
            return "\0"

        return self.input[self.index]

    def peek(self):
        try:
            return self.input[self.index + 1]
        except IndexError:
            return "\0"

    def mark(self):
        self.marks.append(self.index)
        return self.index

    def pop(self):
        self.marks.pop()
        return self.index

    def reset(self):
        self.index = self.marks.pop()
        return self.index

    @property
    def start(self):
        return self.index == 0

    @property
    def end(self):
        return self.index >= len(self.input)

    def __str__(self):
        return f"Consumer(input='{self.input}', index={self.index}, marks={self.marks}, remaining={self.input[self.index:]})"


class Matcher(abc.ABC):

    @abc.abstractmethod
    def test(self, input: str):
        pass


@dataclasses.dataclass
class Literal(Matcher):

    value: str

    def test(self, input: Consumer):
        for character in self.value:
            if input.next() != character:
                return False

        return True


@dataclasses.dataclass
class Range(Matcher):

    character_class: CharacterClass

    def test(self, input: Consumer):
        return self.character_class.test(input.next())


@dataclasses.dataclass
class Group(Matcher):

    values: str
    negate: bool = False

    def test(self, input: Consumer):
        next = input.next()
        if next == "\0":
            return False

        if self.negate:
            return next not in self.values
        else:
            return next in self.values


@dataclasses.dataclass
class Start(Matcher):

    def test(self, input: Consumer):
        return input.start


@dataclasses.dataclass
class End(Matcher):

    def test(self, input: Consumer):
        return input.end


@dataclasses.dataclass
class Repeat(Matcher):

    delegate: Matcher
    min: int
    max: int = 0xffffffff

    def test(self, input: Consumer):
        for x in range(self.max):
            input.mark()
            if not self.delegate.test(input):
                input.reset()
                return x >= self.min

            input.pop()

        return True


@dataclasses.dataclass
class Node:

    name: str
    matchers: typing.List[typing.Tuple[Matcher, "Node"]] = dataclasses.field(default_factory=list)

    @property
    def end(self):
        return len(self.matchers) == 0

    def add(self, matcher: Matcher, node: "Node"):
        self.matchers.append((matcher, node))

    def print(self, depth=0):
        tab = "    " * depth

        print(f'{tab}Node "{self.name}"')

        if not self.matchers:
            print(f'{tab}End')
            return

        for index, (matcher, node) in enumerate(self.matchers):
            print(f'{tab}  [{index}] {matcher}')
            node.print(depth + 1)


def build(pattern):
    matchers: typing.List[Matcher] = []

    index = 0

    def consume():
        nonlocal index

        try:
            character = pattern[index]
            index += 1

            return character
        except IndexError:
            return "\0"

    def read_current():
        try:
            return pattern[index]
        except IndexError:
            return "\0"

    def read_peek():
        try:
            return pattern[index + 1]
        except IndexError:
            return "\0"

    while index < len(pattern):
        current = consume()

        match current:
            case '\\':
                klass = consume()

                if klass == '\\':
                    matcher = Literal("\\")
                else:
                    matcher = Range(CharacterClass.of(klass))

                matchers.append(matcher)

            case '^':
                matchers.append(Start())

            case '$':
                matchers.append(End())

            case '[':
                values = ""
                negate = False

                while True:
                    current = consume()

                    if current == '^' and not values:
                        negate = True
                        continue

                    if current == "]":
                        break

                    values += current

                matchers.append(Group(values, negate))

            case '+':
                matcher = matchers.pop()
                matchers.append(Repeat(matcher, 1))

            case '?':
                matcher = matchers.pop()
                matchers.append(Repeat(matcher, 0, 1))

            case _:
                matchers.append(Literal(current))

    start = Node("start")

    node = start
    for index, matcher in enumerate(matchers, 1):
        next = Node(f"q{index}")
        node.add(matcher, next)
        node = next

    return start


def match(root: Node, input: Consumer, is_start=True) -> bool:
    if root.end:
        return True

    while True:
        for matcher, node in root.matchers:
            input.mark()

            if matcher.test(input):
                return match(node, input, is_start=False)

            input.reset()

        if not is_start or input.end:
            break

        input.next()

    return False


def main():
    pattern = sys.argv[2]
    input_line = sys.stdin.read()

    if sys.argv[1] != "-E":
        print("Expected first argument to be '-E'")
        exit(1)

    graph = build(pattern)
    graph.print()

    if match(graph, Consumer(input_line)):
        exit(0)
    else:
        exit(1)


if __name__ == "__main__":
    main()
