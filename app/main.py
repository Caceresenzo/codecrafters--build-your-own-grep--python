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
    max: int = -1
    marks: typing.List[typing.Tuple[int, int]] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        self.max = len(self.input)

    def next(self):
        if self.end:
            return "\0"

        previous_index = self.index
        self.index += 1

        return self.input[:self.max][previous_index]

    def current(self):
        if self.end:
            return "\0"

        return self.input[:self.max][self.index]

    def peek(self):
        try:
            return self.input[:self.max][self.index + 1]
        except IndexError:
            return "\0"

    def mark(self, *, index=None, max=None):
        # print(f"mark, marks={self.marks}")
        self.marks.append((self.index, self.max))

        if index is not None:
            if index < 0:
                self.index -= index
            else:
                self.index = index

        if max is not None:
            if max < 0:
                self.max -= max
            else:
                self.max = max

    def pop(self):
        # print(f"pop, marks={self.marks}")
        self._pop()

    def _pop(self):
        if len(self.marks):
            return self.marks.pop()

        if self.index != 0 or self.max != len(self.input):
            return 0, len(self.input)

        raise ValueError("pop stack empty")

    def reset(self, *, index=True, max=True):
        # print(f"reset, marks={self.marks}")
        old_index, old_max = self._pop()

        if index:
            self.index = old_index

        if max:
            self.max = old_max

    def restore(self, *, index=False, max=False):
        old_index, old_max = self.marks[-1]

        if index:
            self.index = old_index

        if max:
            self.max = old_max

    def slice(self, start: int):
        return self.input[:self.max][start:self.index]

    @property
    def start(self):
        return self.index == 0

    @property
    def end(self):
        return self.index >= self.max

    def __str__(self):
        return f"Consumer('{self.input}'[{self.index}:{self.max}]='{self.input[self.index:self.max]}', marks={self.marks})"


@dataclasses.dataclass(kw_only=True)
class Node(abc.ABC):

    name: str = "unnamed"

    def match(self, input: Consumer) -> bool:
        raise NotImplementedError()

    def add(self, node: "Node"):
        self.children.append(node)


@dataclasses.dataclass(kw_only=True)
class MatchNode(Node):

    def test(self, input):
        return True

    def match(self, input):
        input.mark()

        if self.test(input):
            input.pop()
            return True

        input.reset()
        return False


@dataclasses.dataclass(kw_only=True)
class Wildcard(MatchNode):

    def test(self, input):
        return input.next() != "\0"


@dataclasses.dataclass(kw_only=True)
class Literal(MatchNode):

    value: str

    def test(self, input):
        for character in self.value:
            if input.next() != character:
                return False

        return True


@dataclasses.dataclass(kw_only=True)
class Range(MatchNode):

    character_class: CharacterClass

    def test(self, input):
        return self.character_class.test(input.next())


@dataclasses.dataclass(kw_only=True)
class Group(MatchNode):

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
class StartAnchor(Node):

    def match(self, input):
        return input.start


@dataclasses.dataclass(kw_only=True)
class EndAnchor(Node):

    def match(self, input):
        return input.end


@dataclasses.dataclass(kw_only=True)
class Repeat(Node):

    min: int
    max: int = 0xffffffff
    child: "Node" = dataclasses.field(repr=False)

    def match(self, input):
        for x in range(self.max):
            input.mark()

            if not self.child.match(input):
                input.reset()
                return x >= self.min

            input.pop()

        return True


@dataclasses.dataclass(kw_only=True)
class Capture(Node):

    number: int
    value: str = None
    child: "Node" = dataclasses.field(repr=False)

    def match(self, input):
        start = input.index

        if self.child.match(input):
            self.value = input.slice(start)
            return True

        return False


@dataclasses.dataclass(kw_only=True)
class Backreference(MatchNode):

    number: int
    capture: Capture

    def test(self, input):
        delegate = Literal(value=self.capture.value)

        return delegate.test(input)


@dataclasses.dataclass(kw_only=True)
class And(Node):

    children: typing.List["Node"] = dataclasses.field(repr=False, default_factory=list)

    def match(self, input):
        for node in self.children:
            if not node.match(input):
                return False

        return True


@dataclasses.dataclass(kw_only=True)
class Or(Node):

    children: typing.List["Node"] = dataclasses.field(repr=False, default_factory=list)

    def match(self, input):
        for node in self.children:
            if node.match(input):
                return True

        return False


@dataclasses.dataclass(kw_only=True)
class Final(Node):

    name = "final"

    def match(self, input):
        return True


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

    def parse():
        nonlocal capture_number

        ors: typing.List[Or] = []
        nodes: typing.List[Node] = []

        def append_repeat(min: int, max=Repeat.max):
            last = nodes.pop()
            if isinstance(last, Literal) and len(last.value) > 1:
                nodes.append(Literal(value=last.value[:-1]))
                last = Literal(value=last.value[-1])

            nodes.append(Repeat(child=last, min=min, max=max))

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
                    nodes.append(StartAnchor())

                case '$':
                    nodes.append(EndAnchor())

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
                    capture = Capture(number=number, child=nested)

                    nodes.append(capture)
                    captures.append(capture)

                case ')':
                    break

                case '|':
                    ors.append(And(children=nodes) if len(nodes) > 1 else nodes[0])
                    nodes = []

                case '.':
                    nodes.append(Wildcard())

                case '+': append_repeat(min=1)
                case '?': append_repeat(min=0, max=1)

                case _:
                    last = nodes[-1] if len(nodes) else None
                    if isinstance(last, Literal):
                        last.value += current
                    else:
                        nodes.append(Literal(value=current))

        if ors:
            if nodes:
                ors.append(And(children=nodes))

            return Or(children=ors)

        return And(children=nodes)

    start = And(name="start", children=[parse()])
    start.children.append(Final())

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

            if hasattr(node, "children"):
                post_process(node.children)

            if hasattr(node, "child"):
                post_process([node.child])

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


def draw(node: Node, depth=0):
    tab = "    " * depth

    print(f'{tab}{node}')

    if isinstance(node, Wildcard):
        pass
    elif isinstance(node, Literal):
        # print(f'{tab}  .value="{node.value}"')
        pass
    elif isinstance(node, Range):
        # print(f'{tab}  .character_class="{node.character_class.name}"')
        pass
    elif isinstance(node, Group):
        # print(f'{tab}  .values="{node.values}"')
        # print(f'{tab}  .negate={node.negate}')
        pass
    elif isinstance(node, StartAnchor):
        pass
    elif isinstance(node, EndAnchor):
        pass
    elif isinstance(node, Repeat):
        # print(f'{tab}  .min={node.min}')
        # print(f'{tab}  .max={node.max}')
        draw(node.child, depth + 1)
    elif isinstance(node, Capture):
        # print(f'{tab}  .number={node.number}')
        # print(f'{tab}  .value="{node.value}"')
        draw(node.child, depth + 1)
    elif isinstance(node, Backreference):
        # print(f'{tab}  .number={node.number}')
        # print(f'{tab}  .capture="{node.capture}"')
        pass
    elif isinstance(node, (And, Or)):
        for child in node.children:
            draw(child, depth + 1)
    elif isinstance(node, Final):
        pass
    else:
        raise NotImplementedError(str(node))


def main():
    pattern = sys.argv[2]
    input_line = sys.stdin.read()

    if sys.argv[1] != "-E":
        print("Expected first argument to be '-E'")
        exit(1)

    graph, captures = build(pattern)
    draw(graph)

    if match(graph, input_line):
        for index, capture in enumerate(captures):
            print(f"group[{index + 1}] = `{capture.value}`")

        exit(0)
    else:
        exit(1)


if __name__ == "__main__":
    main()
