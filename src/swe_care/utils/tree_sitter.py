"""Tree-sitter powered Python stub generator.

This module provides a small utility class, ``TreeSitterStubGenerator``, which
parses Python source code with Tree-sitter, extracts the public structure
(classes, functions, variables) and their docstrings, and renders a simple
``.pyi``-style stub string.

Notes
-----
- Only Python is supported currently.
- We use Tree-sitter queries to locate the definition nodes, then read fields
  directly from AST nodes for signatures and docstrings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Literal, Optional, Tuple, Union

try:
    import tree_sitter_python as tspython
    from tree_sitter import Language, Node, Parser, Query, QueryCursor  # type: ignore
except Exception as exc:  # pragma: no cover - import-time guard
    raise ImportError(
        "tree_sitter and tree_sitter_python are required for this module. "
        "Install with: pip install tree_sitter tree_sitter_python"
    ) from exc


# Initialize Python language for the parser once.
PY_LANGUAGE: Language = Language(tspython.language())


@dataclass(slots=True)
class VariableInfo:
    name: str
    annotation: Optional[str] = None
    # Variables in Python do not have docstrings; kept for parity.
    docstring: Optional[str] = None
    # Internal: byte offsets for stable ordering
    _start_byte: int = 0


@dataclass(slots=True)
class FunctionInfo:
    name: str
    params: str
    returns: Optional[str]
    docstring: Optional[str]
    decorators: List[str] = field(default_factory=list)
    is_async: bool = False
    # Internal: byte offsets for stable ordering
    _start_byte: int = 0


@dataclass(slots=True)
class ClassInfo:
    name: str
    bases: Optional[str]
    docstring: Optional[str]
    decorators: List[str] = field(default_factory=list)
    methods: List[FunctionInfo] = field(default_factory=list)
    # Internal: byte offsets for stable ordering
    _start_byte: int = 0


@dataclass(slots=True)
class ModuleStructure:
    module_docstring: Optional[str]
    classes: List[ClassInfo]
    functions: List[FunctionInfo]
    variables: List[VariableInfo]


class TreeSitterStubGenerator:
    """Generate structured info and stubs from Python code using Tree-sitter.

    Parameters
    ----------
    language: Literal["Python"]
        Currently only "Python" is supported. The parameter exists to
        anticipate future multi-language support.
    """

    def __init__(self, language: Literal["Python"] = "Python") -> None:
        if language != "Python":
            raise ValueError("Only Python is supported at the moment.")
        self.language = PY_LANGUAGE
        self.parser: Parser = Parser(self.language)

    # ------------------------------ Public API ------------------------------
    def extract_structure(self, code: str) -> ModuleStructure:
        """Extract module structure (classes, functions, variables, docstrings).

        This method relies on Tree-sitter queries to locate definitions, then
        reads relevant fields from the matched nodes to build a structured view
        usable for stub generation.

        Parameters
        ----------
        code: str
            Python source code to analyze.

        Returns
        -------
        ModuleStructure
            Extracted structure including classes, functions, variables, and
            docstrings.
        """

        source_bytes = code.encode("utf8")
        tree = self.parser.parse(source_bytes)
        root = tree.root_node

        module_docstring = self._extract_docstring_from_module(root, source_bytes)

        # Use queries to obtain top-level definitions (function/class) and
        # assignments. We purposefully capture the definition node itself and
        # then read structured fields from the node, which is resilient across
        # tree-sitter Python grammar versions.

        # Function definitions (including decorated ones)
        fn_query = Query(
            self.language,
            """
            (function_definition) @func.def
            (decorated_definition
                definition: (function_definition) @func.def
            )
            """,
        )

        # Class definitions (including decorated ones)
        cls_query = Query(
            self.language,
            """
            (class_definition) @class.def
            (decorated_definition
                definition: (class_definition) @class.def
            )
            """,
        )

        # Assignments at any level (we will post-filter to module-level only).
        # Some grammar versions may not have a separate `annotated_assignment`
        # node; limit query to `assignment` and later detect annotations.
        var_query = Query(
            self.language,
            """
            (assignment) @var.assign
            """,
        )

        # Collect nodes using a cursor-based helper to handle different
        # tree-sitter Python bindings across environments.
        func_nodes = [
            node
            for node, name in self._query_captures(fn_query, root, source_bytes)
            if name == "func.def"
        ]
        class_nodes = [
            node
            for node, name in self._query_captures(cls_query, root, source_bytes)
            if name == "class.def"
        ]
        var_nodes = [
            node
            for node, name in self._query_captures(var_query, root, source_bytes)
            if name == "var.assign"
        ]

        # Map functions to enclosing classes (if any), otherwise module level.
        functions_by_class_key: dict[Tuple[int, int], list[FunctionInfo]] = {}
        module_functions: list[FunctionInfo] = []

        for fn_node in func_nodes:
            enclosing_class = self._find_enclosing_class(fn_node)
            fn_info = self._build_function_info(fn_node, source_bytes)
            if fn_info is None:
                continue
            if enclosing_class is None:
                if self._is_public_name(fn_info.name):
                    module_functions.append(fn_info)
            else:
                class_key = (enclosing_class.start_byte, enclosing_class.end_byte)
                functions_by_class_key.setdefault(class_key, []).append(fn_info)

        # Build class infos, attaching their methods (sorted by appearance)
        class_infos: list[ClassInfo] = []
        for cls_node in class_nodes:
            cls_info = self._build_class_info(
                cls_node, source_bytes, functions_by_class_key
            )
            if cls_info is None:
                continue
            if self._is_public_name(cls_info.name):
                class_infos.append(cls_info)

        # Identify module-level variables from assignment nodes
        var_infos: list[VariableInfo] = []
        for v_node in var_nodes:
            if not self._is_module_level(v_node):
                continue
            var_infos.extend(self._build_variable_infos(v_node, source_bytes))

        # Stable ordering by source position
        class_infos.sort(key=lambda c: c._start_byte)
        module_functions.sort(key=lambda f: f._start_byte)
        var_infos.sort(key=lambda v: v._start_byte)

        return ModuleStructure(
            module_docstring=module_docstring,
            classes=class_infos,
            functions=module_functions,
            variables=[
                v
                for v in var_infos
                if self._is_public_name(v.name) or v.name == "__all__"
            ],
        )

    def generate_stub(self, code_or_structure: Union[str, ModuleStructure]) -> str:
        """Render a simple .pyi-style stub for given code or structure.

        If provided a ``str`` of Python source code, this method parses and
        extracts structure internally. If provided a ``ModuleStructure``, it is
        rendered directly.
        """

        structure = (
            self.extract_structure(code_or_structure)
            if isinstance(code_or_structure, str)
            else code_or_structure
        )

        lines: list[str] = []

        need_any = any(v.annotation is None for v in structure.variables)
        if need_any:
            lines.append("from typing import Any")

        if structure.module_docstring:
            if lines:
                lines.append("")
            lines.append(self._render_docstring(structure.module_docstring))

        # Render top-level declarations in original source order.
        # Build a combined, ordered list of (start_byte, kind, object)
        items: list[tuple[int, str, object]] = []
        for v in structure.variables:
            items.append((getattr(v, "_start_byte", 0), "var", v))
        for f in structure.functions:
            items.append((getattr(f, "_start_byte", 0), "func", f))
        for c in structure.classes:
            items.append((getattr(c, "_start_byte", 0), "class", c))

        # Sort by starting byte; Python's sort is stable for equal positions.
        items.sort(key=lambda t: t[0])

        # Emit them interleaved, preserving occurrence order.
        for _, kind, obj in items:
            if lines:
                lines.append("")
            if kind == "var":
                v = obj  # type: ignore[assignment]
                ann = getattr(v, "annotation", None) or "Any"
                lines.append(f"{getattr(v, 'name')}: {ann} = ...")
            elif kind == "func":
                fn = obj  # type: ignore[assignment]
                lines.extend(self._render_function(fn, indent=0))
            elif kind == "class":
                cls = obj  # type: ignore[assignment]
                lines.extend(self._render_class(cls))

        return "\n".join(lines) + ("\n" if lines else "")

    # ----------------------------- Helper methods ----------------------------
    @staticmethod
    def _indent(s: str, spaces: int) -> str:
        pad = " " * spaces
        return "\n".join(pad + line if line else line for line in s.splitlines())

    def _get_text(self, node: Node, source: bytes) -> str:
        return source[node.start_byte : node.end_byte].decode("utf8")

    def _is_public_name(self, name: str) -> bool:
        # Treat dunder names as private for purposes of public API, except
        # allow __all__ which often signals the intended public surface.
        return not name.startswith("_") and not (
            name.startswith("__") and name.endswith("__")
        )

    def _is_module_level(self, node: Node) -> bool:
        # A node is module-level if it has no class/function/def ancestors.
        cur = node.parent
        while cur is not None:
            if cur.type in {
                "function_definition",
                "class_definition",
                "decorated_definition",
            }:
                # Decorated defs are still definitions and not module-level
                return False
            if cur.type == "module":
                return True
            cur = cur.parent
        return True

    def _find_enclosing_class(self, node: Node) -> Optional[Node]:
        cur = node.parent
        while cur is not None:
            if cur.type == "class_definition":
                return cur
            cur = cur.parent
        return None

    def _extract_docstring_from_module(
        self, root: Node, source: bytes
    ) -> Optional[str]:
        # Python module docstring is the first statement if it is a string
        if root.type != "module" or len(root.children) == 0:
            return None

        first_stmt = root.children[0]
        # May be an expression_statement containing a string literal
        if first_stmt.type == "expression_statement" and first_stmt.children:
            inner = first_stmt.children[0]
            if inner.type in {"string", "concatenated_string"}:
                text = self._get_text(inner, source)
                return self._string_literal_value(text)
        return None

    def _extract_docstring_from_block(
        self, block_node: Optional[Node], source: bytes
    ) -> Optional[str]:
        if block_node is None:
            return None
        # Block body typically starts with an expression_statement (string)
        for child in block_node.children:
            if child.type == "expression_statement" and child.children:
                inner = child.children[0]
                if inner.type in {"string", "concatenated_string"}:
                    text = self._get_text(inner, source)
                    return self._string_literal_value(text)
            # Any non-empty statement before a string means no docstring
            if child.type not in {"comment"} and child.is_named:
                break
        return None

    def _string_literal_value(self, literal_src: str) -> str:
        # Attempt to parse the Python string literal to get its actual value
        # (handles quotes and escapes). Fallback to a naive strip of quotes.
        try:
            import ast

            return ast.literal_eval(literal_src)
        except Exception:
            # naive: strip a single leading/trailing quote set
            s = literal_src.strip()
            if s[:3] in {'"""', "'''"} and s[-3:] == s[:3]:
                return s[3:-3]
            if s and s[0] in {'"', "'"} and s[-1] == s[0]:
                return s[1:-1]
            return s

    def _build_function_info(
        self, fn_node: Node, source: bytes
    ) -> Optional[FunctionInfo]:
        name_node = fn_node.child_by_field_name("name")
        params_node = fn_node.child_by_field_name("parameters")
        body_node = fn_node.child_by_field_name("body")
        returns_node = fn_node.child_by_field_name("return_type")

        if name_node is None or params_node is None:
            return None

        name = self._get_text(name_node, source)
        params = self._get_text(params_node, source)  # includes parentheses
        returns: Optional[str]
        if returns_node is not None:
            returns = self._get_text(returns_node, source)
        else:
            returns = None

        # Determine if async is present (a child token named 'async')
        is_async = any(ch.type == "async" for ch in fn_node.children)

        # Decorators can be provided by a decorated_definition parent
        decorators: list[str] = []
        parent = fn_node.parent
        if parent is not None and parent.type == "decorated_definition":
            for ch in parent.children:
                if ch.type == "decorator":
                    decorators.append(self._get_text(ch, source))

        doc = self._extract_docstring_from_block(body_node, source)

        return FunctionInfo(
            name=name,
            params=params,
            returns=returns,
            docstring=doc,
            decorators=decorators,
            is_async=is_async,
            _start_byte=fn_node.start_byte,
        )

    def _build_class_info(
        self,
        cls_node: Node,
        source: bytes,
        functions_by_class_key: dict[Tuple[int, int], list[FunctionInfo]],
    ) -> Optional[ClassInfo]:
        name_node = cls_node.child_by_field_name("name")
        body_node = cls_node.child_by_field_name("body")
        bases_node = cls_node.child_by_field_name("superclasses")

        if name_node is None or body_node is None:
            return None

        name = self._get_text(name_node, source)
        bases = self._get_text(bases_node, source) if bases_node else None

        # If wrapped in a decorated_definition, collect decorators from parent
        decorators: list[str] = []
        parent = cls_node.parent
        if parent is not None and parent.type == "decorated_definition":
            for ch in parent.children:
                if ch.type == "decorator":
                    decorators.append(self._get_text(ch, source))

        doc = self._extract_docstring_from_block(body_node, source)

        class_key = (cls_node.start_byte, cls_node.end_byte)
        methods = functions_by_class_key.get(class_key, [])

        return ClassInfo(
            name=name,
            bases=bases,
            docstring=doc,
            decorators=decorators,
            methods=methods,
            _start_byte=cls_node.start_byte,
        )

    def _build_variable_infos(self, node: Node, source: bytes) -> list[VariableInfo]:
        infos: list[VariableInfo] = []
        if node.type == "annotated_assignment":
            # Some grammar versions may expose this distinct node type.
            target = node.child_by_field_name("target")
            type_node = node.child_by_field_name("type")
            if target is not None and target.type == "identifier":
                name = self._get_text(target, source)
                ann = self._get_text(type_node, source) if type_node else None
                infos.append(
                    VariableInfo(name=name, annotation=ann, _start_byte=node.start_byte)
                )
            return infos

        if node.type == "assignment":
            left = node.child_by_field_name("left")
            if left is not None:
                # Attempt to detect an annotation inside the target.
                annotation_text = self._find_annotation_text(left, source)
                # Collect identifiers from single name or pattern_list/tuple/list
                for ident in self._iter_identifiers(left):
                    name = self._get_text(ident, source)
                    infos.append(
                        VariableInfo(
                            name=name,
                            annotation=annotation_text,
                            _start_byte=node.start_byte,
                        )
                    )
            return infos

        return infos

    def _iter_identifiers(self, node: Node) -> Iterable[Node]:
        if node.type == "identifier":
            yield node
            return
        # pattern_list, tuple, list, etc.
        for ch in node.children:
            yield from self._iter_identifiers(ch)

    def _query_captures(
        self, query: Query, root: Node, source: bytes
    ) -> list[tuple[Node, str]]:
        """Run a query and return a normalized list of (node, capture_name).

        Tree-sitter Python bindings have changed signatures across versions.
        This helper tries a few call patterns to remain compatible.
        """
        # Try Query.captures(root) if available.
        try:
            out = query.captures(root)  # type: ignore[attr-defined]
            if out is not None:
                result = list(out)
                # If capture name is an int id, map via capture_names
                if result and isinstance(result[0][1], int):
                    names = getattr(query, "capture_names", None) or getattr(
                        query, "captures_names", None
                    )
                    if names:
                        return [(n, names[idx]) for n, idx in result]
                return result  # assume list[(Node, str)]
        except Exception:
            pass

        # Try QueryCursor APIs
        try:
            # Newer style may accept the query in the constructor
            cursor = QueryCursor(query)  # type: ignore[call-arg]
            try:
                out = cursor.captures(root)  # type: ignore[arg-type]
            except TypeError:
                out = cursor.captures(root, source)  # type: ignore[arg-type]
        except TypeError:
            # Older style: create cursor then exec
            cursor = QueryCursor()
            try:
                cursor.exec(query, root)  # type: ignore[attr-defined]
                try:
                    out = cursor.captures(root)  # type: ignore[arg-type]
                except TypeError:
                    out = cursor.captures(root, source)  # type: ignore[arg-type]
            except Exception:
                # Some versions use captures(query, root, ...)
                try:
                    out = cursor.captures(query, root)  # type: ignore[misc]
                except TypeError:
                    out = cursor.captures(query, root, source)  # type: ignore[misc]

        # Normalize output
        if isinstance(out, dict):
            pairs: list[tuple[Node, str]] = []
            for name, nodes in out.items():  # type: ignore[assignment]
                for n in nodes:
                    pairs.append((n, name))
            return pairs

        out_list = list(out) if out is not None else []  # type: ignore[arg-type]
        if out_list and isinstance(out_list[0][1], int):
            names = getattr(query, "capture_names", None) or getattr(
                query, "captures_names", None
            )
            if names:
                return [(n, names[idx]) for n, idx in out_list]
        return out_list  # assume list[(Node, str)]

    def _find_annotation_text(self, node: Node, source: bytes) -> Optional[str]:
        """Heuristically find a type annotation text attached to a target.

        On some grammar versions, annotated assignment is modeled as an
        `assignment` whose left-hand side contains a child node with field
        name `type` (or a node literally named `type`). We search shallowly
        to locate such a node.
        """
        # Prefer a field named "type" if present
        type_field = node.child_by_field_name("type")
        if type_field is not None:
            return self._get_text(type_field, source)
        # Fallback: DFS to any node named exactly "type"
        stack = [node]
        visited: set[int] = set()
        while stack:
            cur = stack.pop()
            if id(cur) in visited:
                continue
            visited.add(id(cur))
            if cur.type == "type":
                return self._get_text(cur, source)
            stack.extend(cur.children)
        return None

    # ----------------------------- Rendering utils ---------------------------
    def _render_docstring(self, text: str) -> str:
        # Normalize indentation to avoid double-indenting when nested.
        import textwrap

        normalized = textwrap.dedent(text).strip("\n")
        # Use triple-double-quotes; escape internal triple-double-quotes.
        safe = normalized.replace('"""', '\\"\\"\\"')
        return f'"""{safe}"""'

    def _render_function(self, fn: FunctionInfo, indent: int) -> list[str]:
        lines: list[str] = []
        for dec in fn.decorators:
            dec_line = dec.strip()
            if not dec_line.startswith("@"):
                dec_line = "@" + dec_line.lstrip("@")
            if indent:
                lines.append(self._indent(dec_line, indent))
            else:
                lines.append(dec_line)

        header = ("async def " if fn.is_async else "def ") + f"{fn.name}{fn.params}"
        if fn.returns:
            header += f" -> {fn.returns}"
        header += ":"
        if indent:
            header = self._indent(header, indent)
        lines.append(header)

        body_lines: list[str] = []
        if fn.docstring:
            body_lines.append(self._render_docstring(fn.docstring))
        body_lines.append("...")

        if indent:
            lines.extend([self._indent(line, indent + 4) for line in body_lines])
        else:
            lines.extend([self._indent(line, 4) for line in body_lines])

        return lines

    def _render_class(self, cls: ClassInfo) -> list[str]:
        """Render a single class block without trailing blank lines."""
        lines: list[str] = []
        header = f"class {cls.name}"
        if cls.bases:
            header += f"{cls.bases}"
        header += ":"
        lines.append(header)

        added_any = False
        if cls.docstring:
            lines.append(self._indent(self._render_docstring(cls.docstring), 4))
            added_any = True

        if cls.methods:
            if added_any:
                lines.append("")
            for method in sorted(cls.methods, key=lambda m: m._start_byte):
                lines.extend(self._render_function(method, indent=4))
                lines.append("")
            if lines and lines[-1] == "":
                lines.pop()
            added_any = True

        # ensure non-empty class body
        if not added_any:
            lines.append(self._indent("...", 4))

        return lines
