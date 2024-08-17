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
        is_in = input.next() in self.values

        if self.negate:
            is_in = not is_in

        return is_in


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
    start = Node("start")
    node_count = 0

    current_node = start

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

    def link(matcher: Matcher, node: Node = None):
        nonlocal current_node, node_count

        next = Node(f"q{node_count}")
        node_count += 1

        if node is None:
            node = current_node

        node.add(matcher, next)
        current_node = next

    while index < len(pattern):
        current = consume()
        print(f"current `{current}`  {index}")

        match current:
            case '\\':
                klass = consume()

                matcher = Range(CharacterClass.of(klass))
                link(matcher)

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

                matcher = Group(values, negate)
                link(matcher)

            case _:
                value = current

                while True:
                    next = read_current()
                    if next in "\\[]\0":
                        break

                    value += next
                    index += 1

                matcher = Literal(value)
                link(matcher)

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
    graph.print()

    if match(graph, Consumer(input_line)):
        exit(0)
    else:
        exit(1)


if __name__ == "__main__":
    main()
