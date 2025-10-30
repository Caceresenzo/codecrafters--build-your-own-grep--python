import abc
import enum
from typing import Callable


class Pattern:

    def __init__(self, expression: str, root: "Node", group_count: int):
        self.expression = expression
        self.root = root
        self.group_count = group_count

    def matcher(self, text: str) -> "Matcher":
        return Matcher(self, text)

    @staticmethod
    def compile(expression: str) -> "Pattern":
        parser = PatternParser(expression)
        return parser.parse()


class Matcher:

    def __init__(self, pattern: Pattern, text: str):
        self.pattern = pattern
        self.text = text

        group_count_with_zero = pattern.group_count + 1
        self.group_starts = [0] * group_count_with_zero
        self.group_ends = [0] * group_count_with_zero

        self.reset()
        self.hit_end = False

    def reset(self):
        self.first = -1
        self.last = 0

        self.from_ = 0
        self.to = len(self.text)

    def find(self, /, from_: int) -> bool:
        self.reset()

        return self._search(from_)

    def group(self, /, number=0) -> str:
        start = self.group_starts[number]
        end = self.group_ends[number]

        return self.text[start:end]

    def group_count(self) -> int:
        return self.pattern.group_count

    def _search(self, /, from_: int) -> bool:
        self.hit_end = False

        found = self.pattern.root.match(self, from_, self.text)
        if not found:
            self.first = -1

        return found


class PatternParser:

    def __init__(self, expression: str):
        self.expression = expression
        self.index = 0
        self.group_count = 0

        self.context = self.Context()

    def parse(self) -> Pattern:
        contexts = [self.context]

        while self.has_next():
            if self.match('|'):
                self.context = self.Context()
                contexts.append(self.context)

                # TODO Handle double pipes, aka zero-length
                continue

            self.parse_next()

        absolute_last = LastNode()
        root = StartNode()
        root.next = self.to_branch_if_necessary(contexts, absolute_last, absolute_last)

        return Pattern(self.expression, root, self.group_count)

    def parse_next(self):
        character = self.consume()

        match character:
            case '\\':
                self.handle_escape()

            case '[':
                self.handle_character_group()

            case '^':
                self.context.add(BeginNode())

            case '$':
                self.context.add(EndNode())

            case '+':
                raise ValueError("unescaped `+` is not allowed")

            case '?':
                raise ValueError("unescaped `?` is not allowed")

            case '.':
                self.handle_character(CharPredicates.Any())

            case '(':
                self.handle_capture_group()

            case _:
                self.handle_character(CharPredicates.Character(character))

    def has_next(self) -> bool:
        return self.index < len(self.expression)

    def peek(self) -> str:
        if self.index >= len(self.expression):
            return '\0'

        return self.expression[self.index]

    def match(self, character: str) -> bool:
        if self.peek() == character:
            self.consume()
            return True

        return False

    def consume(self) -> str:
        character = self.expression[self.index]
        self.index += 1
        return character

    def handle_character(self, predicate: "CharPredicate"):
        node = CharNode(predicate)
        self.context.add(node)

        quantifier = self.match_quantifier()
        if quantifier is not None:
            node.next = LastNode()

            repeat = RepeatNode(atom=node, min=quantifier[0], max=quantifier[1])
            self.context.replace(repeat)

    def handle_escape(self):
        character = self.consume()

        if character == '\\':
            self.handle_character(CharPredicates.Character(character))
        elif character.isdigit():
            self.context.add(BackReferenceNode(int(character)))
        else:
            klass = CharacterRangeClass.of(character)
            self.handle_character(CharPredicates.RangeClass(klass))

    def handle_character_group(self):
        array = CharPredicates.Array()
        ranges: list[CharPredicates.RangeClass] = []

        negate = self.match('^')

        while self.has_next():
            character = self.consume()

            if character == ']':
                break

            if character == '\\':
                character = self.consume()

                if character == '\\':
                    array.add(character)
                else:
                    klass = CharacterRangeClass.of(character)
                    ranges.append(CharPredicates.RangeClass(klass))
            else:
                array.add(character)

        if len(ranges):
            predicate = CharPredicates.Or(array, *ranges)
        else:
            predicate = array

        if negate:
            predicate = CharPredicates.Not(predicate)

        self.handle_character(predicate)

    def handle_capture_group(self):
        self.group_count += 1
        number = self.group_count

        previous_context = self.context

        self.context = self.Context()
        contexts = [self.context]  # TODO handle branches

        while not self.match(')'):
            if self.match('|'):
                self.context = self.Context()
                contexts.append(self.context)

                # TODO Handle consecutive pipes, as context.root will be null
                continue

            self.parse_next()

        tail = GroupHeadTailNode(number)
        head = GroupHeadNode(number, tail)

        root = self.to_branch_if_necessary(contexts, tail, LastNode())
        head.next = root

        self.context = previous_context

        quantifier = self.match_quantifier()
        if quantifier is not None:
            repeat = RepeatNode(atom=head, min=quantifier[0], max=quantifier[1])
            tail.next = LastNode()

            self.context.add(repeat)
        else:
            self.context.add2(head, tail)

    def match_quantifier(self):
        if self.match('+'):
            return (1, RepeatNode.UNBOUNDED)
        elif self.match('?'):
            return (0, 1)
        elif self.match('*'):
            return (0, RepeatNode.UNBOUNDED)
        elif not self.match('{'):
            return None
        
        times = self.parse_number()

        if not self.match('}'):
            raise ValueError("expected `}` after `{`")
        
        return (times, times)
    
    def parse_number(self) -> int:
        digits = self.consume_while(str.isdigit)

        return int(digits)

    def consume_while(self, predicate: Callable[[str], bool]) -> str:
        builder: str = ""

        while True:
            character = self.peek()
            if character == '\0':
                break

            if not predicate(character):
                break

            builder += self.consume()

        return builder

    def to_branch_if_necessary(self, contexts: list["Context"], last: "Node", intermediate_last: "Node") -> "Node":
        if len(contexts) == 1:
            self.context.end(last, intermediate_last)

            return contexts[0].root

        else:
            roots: list[Node] = []

            for context in contexts:
                context.end(intermediate_last, intermediate_last)
                roots.append(context.root)

            branch = Branch(roots)
            branch.next = last

            return branch

    class Context:

        def __init__(self):
            self.root: "Node" | None = None
            self.current: "Node" | None = None
            self.previous: "Node" | None = None

            self.to_link_to_end: list["Node"] = []

        def add(self, node: "Node"):
            if self.root is None:
                self.root = node
                self.current = node
            else:
                self.previous = self.current
                self.previous.next = self.current = node

        def add2(self, head: "Node", tail: "Node"):
            if self.root is None:
                self.root = head
                self.current = tail
            else:
                self.previous = self.current
                self.previous.next = head
                self.current = tail

        def replace(self, node: "Node"):
            self.to_link_to_end.append(self.current)

            if self.previous is None:
                self.root = self.current = node
            else:
                self.previous.next = self.current = node

        def end(self, last: "Node", intermediate_last: "Node"):
            for node in self.to_link_to_end:
                node.next = intermediate_last

            self.current.next = last


class Node(abc.ABC):

    def __init__(self):
        self.next: Node | None = None

    @abc.abstractmethod
    def match(self, matcher: Matcher, index: int, sequence: str) -> bool:
        ...


class StartNode(Node):

    def match(self, matcher, index, sequence):
        to = matcher.to

        while index < to:
            if self.next.match(matcher, index, sequence):
                matcher.first = 0
                matcher.group_starts[0] = 0
                matcher.group_ends[0] = matcher.last
                return True

            index += 1

        matcher.hit_end = True
        return False

    def __str__(self):
        return "-STAR-"


class CharNode(Node):

    def __init__(self, predicate: "CharPredicate"):
        super().__init__()

        self.predicate = predicate

    def match(self, matcher, index, sequence):
        if index >= matcher.to:
            matcher.hit_end = True
            return False

        character = sequence[index]

        if not self.predicate.test(character):
            return False

        return self.next.match(matcher, index + 1, sequence)

    def __str__(self):
        return str(self.predicate)


class BeginNode(Node):

    def match(self, matcher, index, sequence):
        start_index = matcher.from_

        if index != start_index:
            return False

        if not self.next.match(matcher, index, sequence):
            return False

        matcher.first = start_index
        return True

    def __str__(self):
        return "^"


class EndNode(Node):

    def match(self, matcher, index, sequence):
        end_index = matcher.to

        if index != end_index:
            return False

        if not self.next.match(matcher, index, sequence):
            return False

        return True

    def __str__(self):
        return "$"


class RepeatNode(Node):

    UNBOUNDED = -1

    def __init__(self, *, atom: Node, min: int, max: int):
        super().__init__()

        self.atom = atom
        self.min = min
        self.max = max

    def match(self, matcher, index, sequence):
        count = 0

        while count < self.min:
            if not self.atom.match(matcher, index, sequence):
                return False

            count += 1
            index = matcher.last

        max_count = self.max if self.max != self.UNBOUNDED else 0xffffffff

        return self.greedy_match(matcher, index, sequence, count, max_count)

    def greedy_match(self, matcher: Matcher, index: int, sequence: str, count: int, max_count: int):
        if index < matcher.to and count < max_count and self.atom.match(matcher, index, sequence):
            count += 1
            last_index = matcher.last

            if self.greedy_match(matcher, last_index, sequence, count, max_count):
                return True

            if self.next.match(matcher, last_index, sequence):
                return True

        return self.next.match(matcher, index, sequence)

    def __str__(self):
        min = self.min if self.min != self.UNBOUNDED else ""
        max = self.max if self.max != self.UNBOUNDED else ""
        return f"{{{min},{max}}}"


class Branch(Node):

    def __init__(self, atoms: list[Node]):
        super().__init__()

        self.atoms = atoms

    def match(self, matcher, index, sequence):
        for node in self.atoms:
            if node.match(matcher, index, sequence):
                end_index = matcher.last

                return self.next.match(matcher, end_index, sequence)

        return False

    def __str__(self):
        return "|".join(
            str(atom)
            for atom in self.atoms
        )


class GroupHeadNode(Node):

    def __init__(self, number: int, tail: "GroupHeadTailNode"):
        super().__init__()

        self.number = number
        self.tail = tail

    def match(self, matcher, index, sequence):
        matcher.group_starts[self.number] = index

        if not self.next.match(matcher, index, sequence):
            matcher.group_starts[self.number] = -1
            return False

        return True

    def __str__(self):
        return f"(#{self.number}:"


class GroupHeadTailNode(Node):

    def __init__(self, number: int):
        super().__init__()

        self.number = number

    def match(self, matcher, index, sequence):
        matcher.group_ends[self.number] = index

        if not self.next.match(matcher, index, sequence):
            matcher.group_ends[self.number] = -1
            return False

        return True

    def __str__(self):
        return f":#{self.number})"


class BackReferenceNode(Node):

    def __init__(self, group_number: int):
        super().__init__()

        self.group_number = group_number

    def match(self, matcher, index, sequence):
        start = matcher.group_starts[self.group_number]
        end = matcher.group_ends[self.group_number]

        length = end - start

        # group not matched
        if length < 0:
            return False

        # not enough characters left
        if index + length > matcher.to:
            matcher.hit_end = True
            return False

        for jndex in range(length):
            if sequence[start + jndex] != sequence[index + jndex]:
                return False

        return self.next.match(matcher, index + length, sequence)

    def __str__(self):
        return f"\\{self.group_number}"


class LastNode(Node):

    def match(self, matcher, index, sequence):
        matcher.last = index
        return True

    def __str__(self):
        return "-LAST-"


class CharPredicate(abc.ABC):

    @abc.abstractmethod
    def test(self, character: str) -> bool:
        ...


class CharPredicates:

    def __init__(self):
        raise NotImplementedError("This utility class is not meant to be instantiated directly.")

    class Character(CharPredicate):

        def __init__(self, value: str):
            self.value = value

        def test(self, character):
            return self.value == character

        def __str__(self):
            if self.value == '\\':
                return "\\\\"

            return self.value

    class Or(CharPredicate):

        def __init__(self, first: CharPredicate, *predicates: list[CharPredicate]):
            self.children = [first, *predicates]

        def test(self, character):
            return any(
                child.test(character)
                for child in self.children
            )

        def __str__(self):
            middle = "".join(
                str(child)
                for child in self.children
            )

            return f"[{middle}]"

    class Not(CharPredicate):

        def __init__(self, predicate: CharPredicate):
            self.predicate = predicate

        def test(self, character):
            return not self.predicate.test(character)

        def __str__(self):
            if isinstance(self.predicate, CharPredicates.Or):
                return "[^" + str(self.predicate)[1:]

            return f"[^{self.predicate}]"

    class Any(CharPredicate):

        def test(self, character):
            return True

        def __str__(self):
            return "."

    class Array(CharPredicate):

        def __init__(self):
            self.characters = set()

        def add(self, character: str):
            self.characters.add(character)

        def test(self, character):
            return character in self.characters

        def __str__(self):
            return ",".join(sorted(self.characters))

    class RangeClass(CharPredicate):

        def __init__(self, klass: "CharacterRangeClass"):
            self.klass = klass

        def test(self, character):
            character = ord(character)

            match self.klass:
                case CharacterRangeClass.DIGITS:
                    return character >= ord('0') and character <= ord('9')

                case CharacterRangeClass.WORDS:
                    return (
                        (character >= ord('0') and character <= ord('9'))
                        or character >= ord('A') and character <= ord('Z')
                        or character >= ord('a') and character <= ord('z')
                        or character == ord('_')
                    )

                case _:
                    raise RuntimeError(f"unknown enum: {self}")

        def __str__(self):
            return f"\\{self.klass.value}"


class CharacterRangeClass(enum.Enum):

    DIGITS = "d"
    WORDS = "w"

    @staticmethod
    def of(character: str):
        for klass in CharacterRangeClass:
            if klass.value == character:
                return klass

        raise ValueError(f"unknown range class: {character}")
