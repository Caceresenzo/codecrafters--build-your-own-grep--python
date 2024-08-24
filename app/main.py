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

    def slice(self, start: int):
        return self.input[start:self.index]

    @property
    def start(self):
        return self.index == 0

    @property
    def end(self):
        return self.index >= len(self.input)

    def __str__(self):
        return f"Consumer(input='{self.input}', index={self.index}, marks={self.marks}, remaining={self.input[self.index:]})"


@dataclasses.dataclass(kw_only=True)
class Node(abc.ABC):

    name: str = "unnamed"
    children: typing.List["Node"] = dataclasses.field(default_factory=list, repr=False)
    final: bool = dataclasses.field(default=False, repr=False)

    def test(self, input: Consumer) -> bool:
        return True

    def match(self, input: Consumer) -> bool:
        input.mark()

        if not self.test(input):
            input.reset()
            return False

        for node in self.children:
            result = node.match(input)

            if not result:
                input.reset()
                return False

        return True

    def add(self, node: "Node"):
        self.children.append(node)

    def print(self, depth=0):
        tab = "    " * depth

        print(f'{tab}{self}')

        for node in self.children:
            node.print(depth + 1)

        if self.final:
            print(f'{tab}-- Final --')
            return


@dataclasses.dataclass(kw_only=True)
class Wildcard(Node):

    def test(self, input):
        input.next()
        return True


@dataclasses.dataclass(kw_only=True)
class Literal(Node):

    value: str

    def test(self, input):
        for character in self.value:
            if input.next() != character:
                return False

        return True


@dataclasses.dataclass(kw_only=True)
class Range(Node):

    character_class: CharacterClass

    def test(self, input):
        return self.character_class.test(input.next())


@dataclasses.dataclass(kw_only=True)
class Group(Node):

    values: str
    negate: bool = False

    def test(self, input):
        next = input.next()
        if next == "\0":
            return False

        if self.negate:
            return next not in self.values
        else:
            return next in self.values


@dataclasses.dataclass(kw_only=True)
class Start(Node):

    def test(self, input):
        return input.start


@dataclasses.dataclass(kw_only=True)
class End(Node):

    def test(self, input):
        return input.end


@dataclasses.dataclass(kw_only=True)
class Repeat(Node):

    min: int
    max: int = 0xffffffff

    def match(self, input):
        first, = self.children

        for x in range(self.max):
            input.mark()

            if not first.match(input):
                input.reset()
                return x >= self.min

            input.pop()

        return True


@dataclasses.dataclass(kw_only=True)
class Capture(Node):

    number: int
    value: str = None
    
    def match(self, input):
        start = input.index

        if super().match(input):
            self.value = input.slice(start)
            return True

        return False


@dataclasses.dataclass(kw_only=True)
class Or(Node):

    def match(self, input):
        for node in self.children:
            input.mark()

            if node.match(input):
                return True
    
            input.reset()

        return False


@dataclasses.dataclass(kw_only=True)
class Backreference(Node):

    number: int
    capture: Capture

    def test(self, input):
        delegate = Literal(value=self.capture.value)

        return delegate.test(input)


def build(pattern):
    captures: typing.List[Capture] = []
    capture_number = 0

    q_increment = 0
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

    def parse():
        nonlocal capture_number

        ors: typing.List[Or] = []
        nodes: typing.List[Node] = []

        while index < len(pattern):
            current = consume()

            match current:
                case '\\':
                    klass = consume()

                    if klass.isnumeric():
                        number = int(klass)
                        node = Backreference(number=number, capture=None)
                    elif klass == '\\':
                        node = Literal(value="\\")
                    else:
                        node = Range(character_class=CharacterClass.of(klass))

                    nodes.append(node)

                case '^':
                    nodes.append(Start())

                case '$':
                    nodes.append(End())

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

                    nodes.append(Group(values=values, negate=negate))

                case '(':
                    capture_number += 1
                    number = capture_number

                    nested = parse()
                    capture = Capture(number=number, children=nested)

                    nodes.append(capture)
                    captures.append(capture)

                case ')':
                    break

                case '|':
                    ors.append(Node(children=nodes))
                    nodes = []

                case '.':
                    nodes.append(Wildcard())

                case '+':
                    last = nodes.pop()
                    nodes.append(Repeat(children=[last], min=1))

                case '?':
                    last = nodes.pop()
                    nodes.append(Repeat(children=[last], min=0, max=1))

                case _:
                    nodes.append(Literal(value=current))

        if ors:
            if nodes:
                ors.append(Node(children=nodes))

            return [Or(children=ors)]

        return nodes

    start = Node()
    start.name = "start"
    start.children = parse()
    start.children[-1].final = True

    capture_by_number = {
        capture.number: capture
        for capture in captures
    }

    def post_process(nodes: typing.List[Node]):
        nonlocal q_increment

        for node in nodes:
            if isinstance(node, Backreference):
                node.capture = capture_by_number[node.number]

            q_increment += 1
            node.name = f"q{q_increment}"

            post_process(node.children)
    
    post_process(start.children)

    return start, captures


def match(root: Node, input: str):
    input = Consumer(input)

    while not input.end:
        input.mark()

        if root.match(input):
            return True

        input.reset()
        input.next()


def main():
    pattern = sys.argv[2]
    input_line = sys.stdin.read()

    if sys.argv[1] != "-E":
        print("Expected first argument to be '-E'")
        exit(1)

    graph, captures = build(pattern)
    graph.print()

    if match(graph, input_line):
        for index, capture in enumerate(captures):
            print(f"group[{index + 1}] = `{capture.value}`")

        exit(0)
    else:
        exit(1)


if __name__ == "__main__":
    main()
