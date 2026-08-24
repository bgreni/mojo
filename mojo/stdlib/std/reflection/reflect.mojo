# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Provides the unified `reflect[T]` / `Reflected[T]` reflection API.

`reflect[T]` is a `comptime` alias for the `Reflected[T]` handle type, which
exposes type introspection through static methods. The handle has no runtime
state — `T` is carried entirely in the compile-time parameter — so all queries
are spelled as `reflect[T].method()` (no parens after `[T]`).

- `is_struct()` - whether `T` is a Mojo struct type.
- `field_count()` - number of fields.
- `field_names()` - `Array[StaticString, N]` of field names.
- `field_types()` - a `TypeList` of field types.
- `field_index[name]()` - index of the named field.
- `field[name]` - `Reflected[FieldT]` for the named field's type.
- `field_at[idx]` - `Reflected[FieldT]` for the field at index `idx`.
- `field_offset[name=...]()` / `field_offset[index=...]()` - byte offset.
- `field_ref[idx](s)` - reference to field at index `idx` in value `s`.
- `decorator_types()` - a `TypeList` of the struct's decorator types.
- `decorator_of[D]()` - the struct's decorator value of type `D`, if present.
  Rejected at compile time when `D` conforms to `RepeatableDecorator`.
- `decorators_of[D]()` - every `D` decorator on the struct, nearest-first, for
  a `D` conforming to `RepeatableDecorator`.
- `has_decorator[D]()` - whether the struct carries a `D` decorator.
- `member[name]` - `Member` handle on the named field's *declaration*.
- `member_at[idx]` - `Member` handle on the field *declaration* at index `idx`.

Note the distinction between `field_at[idx]`, a handle on the field's *type*,
and `member_at[idx]`, a handle on the field *declaration* itself -- including
its decorators. `reflect[T].field_at[0]` is the type of the first field;
`reflect[T].member_at[0]` is the field declaration, from which `.type` reaches
the same `Reflected[FieldT]` that `field_at[0]` gives directly.

`reflect` is auto-imported via the prelude, so it is available without
an explicit import. `Reflected[T]` and `Member[Owner, idx]` must be imported
from `std.reflection` when named in signatures.

Example:

```mojo
struct Point:
    var x: Int
    var y: Float64

def print_fields[T: AnyType]():
    comptime names = reflect[T].field_names()
    comptime for i in range(reflect[T].field_count()):
        print(materialize[names[i]]())

def main():
    print_fields[Point]()
```

The wrapped type is exposed as the `T` parameter, so the result of
`field[name]` can be used as a type directly:

```mojo
def main():
    comptime y_type = reflect[Point].field["y"]
    var v: y_type.T = 3.14  # y_type.T is Float64
```
"""

from std.builtin.variadics import ParameterList, TypeList
from std.sys.info import _TargetType, _current_target


trait Decorator:
    """Marker trait for structs that can be used as decorators.

    A struct conforming to `Decorator` may be applied as `@name` or
    `@name(args)` to a `struct` declaration or to a `var` field inside a
    struct body. The parenthesized arguments are constructor arguments, and
    payload is carried by fields. The resulting comptime *value* is recorded
    on the declaration and exposed through `reflect[T].decorator_of[D]()`.
    Field-level decorators are retrieved through the `Member` handle, via
    `reflect[T].member[name]` or `reflect[T].member_at[idx]`.

    At most one decorator of a given type may appear on a single declaration,
    unless that type conforms to `RepeatableDecorator`, which opts out of the
    rule.

    Decorators are inert metadata: they never affect layout, type identity,
    conformances, or code generation.

    A decorator struct may declare **at most one parameter**, which the
    compiler binds to the type of the declaration the decorator is attached
    to: the field's type for a field decorator, the struct's own type for a
    struct decorator. It is never inferred from the constructor arguments and
    cannot be spelled at the use site, so `@field_dec(rename="n")` and
    `@field_dec(skip_if=f)` on the same `var name: String` are both
    `field_dec[String]`. Reading one back means naming that parameterization:
    `reflect[T].member_at[i].decorator_of[field_dec[String]]()`. Two or more
    parameters is an error.

    Keep the parameter's trait bound as weak as the payload allows. It is
    checked at every declaration the decorator is attached to, whatever
    arguments were passed, so an over-strong bound makes the decorator
    unusable on whole categories of field. A payload that only *mentions*
    the parameter in a type -- `Optional[Self.FieldT]` alone, or a function
    over it -- needs nothing, so `AnyType` suffices; storing a *value* of it
    needs `Movable & Deinitable`, taking the argument `var` and transferring
    it with `^`. `ImplicitlyCopyable` is stronger than either and excludes
    `List` and `Dict` fields, which are `Copyable` but not implicitly so.

    Example:
        ```mojo
        struct serde(Decorator):
            var rename: StaticString

            def __init__(out self, rename: StaticString = ""):
                self.rename = rename

        struct Thing:
            @serde(rename="NAME")
            var name: String
        ```

    Example:
        ```mojo
        def _is_empty(s: String) -> Bool:
            return s.byte_length() == 0

        struct field_dec[FieldT: AnyType](Decorator):
            var skip_if: Optional[def (Self.FieldT) thin -> Bool]

            def __init__(
                out self,
                skip_if: Optional[def (Self.FieldT) thin -> Bool] = None,
            ):
                self.skip_if = skip_if

        struct Serialized:
            @field_dec(skip_if=_is_empty)
            var name: String

        def main():
            comptime d = reflect[Serialized].member_at[0].decorator_of[
                field_dec[String]
            ]()
            comptime assert d.value().skip_if.value()("")
        ```
    """

    pass


trait RepeatableDecorator(Decorator):
    """Marker for decorators that may appear more than once on one declaration.

    A `Decorator` may appear at most once per declaration; that is what lets
    `decorator_of[D]()` return an `Optional[D]` that cannot hide a second
    match. A decorator conforming to `RepeatableDecorator` opts out of that
    rule: duplicates are permitted, `decorators_of[D]()` returns all of them
    nearest-to-declaration first, and `decorator_of[D]()` is rejected at
    compile time so no spelling can silently return one of several.

    Example:
        ```mojo
        struct alias_name(RepeatableDecorator):
            var wire: StaticString

            def __init__(out self, wire: StaticString):
                self.wire = wire

        @alias_name("v")
        @alias_name("verbose")
        struct Flag:
            var on: Bool
        ```
    """

    pass


# ===----------------------------------------------------------------------=== #
# Implementation primitives
# ===----------------------------------------------------------------------=== #
#
# These KGEN attributes implement `ContextuallyEvaluatedAttrInterface`, which
# allows them to be evaluated during elaboration after generic type parameters
# have been specialized to concrete types. This is what lets reflection work in
# generic code:
#
#   def foo[T: AnyType]():
#       comptime count = reflect[T].field_count()
# ===----------------------------------------------------------------------=== #


comptime _field_types_of[T: AnyType] = TypeList[
    __mlir_attr[
        `#kgen.struct_field_types<`, T, `> : !kgen.param_list<`, AnyType, `>`
    ]
]

comptime _field_names_of[T: AnyType] = ParameterList[
    __mlir_attr[
        `#kgen.struct_field_names<`, T, `> : !kgen.param_list<!kgen.string>`
    ]
]

comptime _decorator_types_of[T: AnyType] = TypeList[
    __mlir_attr[
        `#kgen.struct_decorator_types<`,
        T,
        `> : !kgen.param_list<`,
        Movable,
        `>`,
    ]
]

comptime _decorator_of[T: AnyType, D: AnyType] = ParameterList[
    __mlir_attr[
        `#kgen.struct_decorator_of<`,
        T,
        `, `,
        D,
        `> : !kgen.param_list<`,
        D,
        `>`,
    ]
]

comptime _decorators_of[T: AnyType, D: AnyType] = ParameterList[
    __mlir_attr[
        `#kgen.struct_decorators_of<`,
        T,
        `, `,
        D,
        `> : !kgen.param_list<`,
        D,
        `>`,
    ]
]

comptime _field_decorator_types_of[T: AnyType, idx: Int] = TypeList[
    __mlir_attr[
        `#kgen.struct_field_decorator_types<`,
        T,
        `, `,
        idx.__mlir_index__(),
        `> : !kgen.param_list<`,
        Movable,
        `>`,
    ]
]

comptime _field_decorator_of[T: AnyType, idx: Int, D: AnyType] = ParameterList[
    __mlir_attr[
        `#kgen.struct_field_decorator_of<`,
        T,
        `, `,
        idx.__mlir_index__(),
        `, `,
        D,
        `> : !kgen.param_list<`,
        D,
        `>`,
    ]
]

comptime _field_decorators_of[T: AnyType, idx: Int, D: AnyType] = ParameterList[
    __mlir_attr[
        `#kgen.struct_field_decorators_of<`,
        T,
        `, `,
        idx.__mlir_index__(),
        `, `,
        D,
        `> : !kgen.param_list<`,
        D,
        `>`,
    ]
]


# ===----------------------------------------------------------------------=== #
# `reflect` / `Reflected[T]`
# ===----------------------------------------------------------------------=== #
#
# `reflect[T]` is a `comptime` alias for `Reflected[T]`, not a function. The
# handle type has no runtime fields — only the compile-time parameter `T` —
# so all introspection answers come from the parameter alone. Every method is
# `@staticmethod`. Spelling access as `reflect[T].method()` (no parens after
# `[T]`) keeps the elaboration-time work elaboration-time and removes the
# zero-sized-instance ceremony the previous instance form required.
# ===----------------------------------------------------------------------=== #


comptime reflect[T: AnyType] = Reflected[T]
"""A compile-time alias for the reflection handle type of `T`.

Resolves to `Reflected[T]`, whose static methods expose introspection of `T`.
Use it as `reflect[T].method()` rather than constructing an instance.

`reflect` is auto-imported via the prelude.

Parameters:
    T: The type to introspect.

Example:
    ```mojo
    struct Point:
        var x: Int
        var y: Float64

    def main():
        print(reflect[Point].field_count())  # 2
    ```
"""


struct Reflected[T: AnyType]:
    """A compile-time reflection handle type for a Mojo type.

    `Reflected[T]` exposes compile-time introspection of `T` through static
    methods. It has no runtime fields — `T` lives entirely in the compile-time
    parameter — and is not constructible. Spell access as `reflect[T].method()`
    (preferred) or `Reflected[T].method()`.

    Member shape — when to use `@staticmethod` vs a `comptime` alias:

    - A member that returns a **type** (e.g. `Reflected[FieldT]`) is a
      `comptime` member alias and is spelled without `()`. This keeps it
      composable in type position: `reflect[T].field["x"].T` reads as a
      type. `field[name]` and `field_at[idx]` are the type-returning
      members today.
    - A member that returns a **value** (an `Int`, `StaticString`,
      `Array`, a `TypeList`, a typed `ref`, etc.) is an
      `@staticmethod` and is spelled with `()` — e.g.
      `reflect[T].field_count()`, `reflect[T].field_names()`,
      `reflect[T].field_index["x"]()`. The `()` at the call site signals
      "evaluate this comptime expression to a value." `field_types`
      returns a `TypeList` value (not a type) and so is a static method.

    When adding a new member: pick a `comptime` alias if the result will be
    used in type position, `@staticmethod` if it will be assigned to a
    `comptime` variable or compared at the call site.

    For best performance, assign the result of static methods that return
    type-level values (such as `field_names`, `field_types`, `field_count`)
    to `comptime` variables so the work happens at compile time.

    Parameters:
        T: The type being introspected. The wrapped type is exposed via this
            parameter, so `reflect[T].T` is `T`.

    Example:
        ```mojo
        struct Point:
            var x: Int
            var y: Float64

        def main():
            comptime if reflect[Point].is_struct():
                comptime names = reflect[Point].field_names()
                comptime for i in range(reflect[Point].field_count()):
                    print(materialize[names[i]]())
        ```
    """

    @staticmethod
    @always_inline("builtin")
    def is_struct() -> Bool:
        """Returns `True` if `T` is a Mojo struct type, `False` otherwise.

        This distinguishes Mojo struct types from MLIR primitive types (such as
        `__mlir_type.index` or `__mlir_type.i64`). The other reflection methods
        produce a compile error on non-struct types, so `is_struct` is useful
        as a `comptime if` guard when iterating over field types that may
        contain MLIR primitives.

        Returns:
            `True` if `T` is a Mojo struct type, `False` if it is an MLIR type.

        Example:
            ```mojo
            def process_type[T: AnyType]():
                comptime if reflect[T].is_struct():
                    print("struct with", reflect[T].field_count(), "fields")
                else:
                    print("non-struct:", reflect[T].name())
            ```
        """
        return __mlir_attr[`#kgen.is_struct_type<`, Self.T, `> : i1`]

    @staticmethod
    def name[*, qualified_builtins: Bool = False]() -> StaticString:
        """Returns the struct name of `T`.

        Parameters:
            qualified_builtins: Whether to print fully qualified builtin type
                names (e.g. `std.builtin.int.Int`) or shorten them
                (e.g. `Int`).

        Returns:
            Type name.

        Example:
            ```mojo
            struct Point:
                var x: Int
                var y: Float64

            def main():
                print(reflect[Point].name())  # "Point" (or module-qualified if defined)
            ```
        """
        return StaticString(
            __mlir_attr[
                `#kgen.get_type_name<`,
                Self.T,
                `, `,
                qualified_builtins._mlir_value,
                `> : !kgen.string`,
            ]
        )

    @staticmethod
    def base_name() -> StaticString:
        """Returns the name of the base type of a parameterized type.

        For parameterized types like `List[Int]`, this returns `"List"`.
        For non-parameterized types, it returns the type's simple name.

        Unlike `name`, this method strips type parameters and returns only the
        unqualified base type name.

        Returns:
            The unqualified name of the base type as a `StaticString`.

        Example:
            ```mojo
            from std.collections import List, Dict

            def main():
                print(reflect[List[Int]].base_name())          # "List"
                print(reflect[Dict[String, Int]].base_name())  # "Dict"
                print(reflect[Int].base_name())                # "Int"
            ```
        """
        return StaticString(
            __mlir_attr[
                `#kgen.get_base_type_name<`,
                Self.T,
                `> : !kgen.string`,
            ]
        )

    @staticmethod
    @always_inline("builtin")
    def field_count() -> Int:
        """Returns the number of fields in struct `T`.

        Constraints:
            `T` must be a struct type.

        Returns:
            The number of fields in the struct.
        """
        return _field_types_of[Self.T]().length

    @staticmethod
    def field_types() -> _field_types_of[Self.T]:
        """Returns the types of all fields in struct `T` as a `TypeList`.

        For nested structs this returns the struct type itself, not its
        flattened fields.

        Constraints:
            `T` must be a struct type.

        Returns:
            A `TypeList` with one entry per field in the struct.

        Example:
            ```mojo
            struct Point:
                var x: Int
                var y: Float64

            def main():
                comptime types = reflect[Point].field_types()
                comptime for i in range(reflect[Point].field_count()):
                    print(reflect[types[i]].name())
            ```
        """
        return {}

    @staticmethod
    def decorator_types() -> _decorator_types_of[Self.T]:
        """Returns the types of the decorators applied to struct `T`.

        A decorator is a struct conforming to `Decorator`, applied as `@name`
        or `@name(args)`. Entries are the decorator types, nearest to the
        declaration first. At most one decorator of a given type may appear
        on a declaration -- unless that type conforms to
        `RepeatableDecorator`, in which case it may appear (and so be
        repeated in this list) more than once. Undecorated structs yield an
        empty list.

        Use this to enumerate what is present; use `decorator_of` to read a
        non-repeatable decorator's value, or `decorators_of` for a repeatable
        one. An enumerate-then-fetch loop that calls `decorator_of[tys[i]]()`
        for every entry hard-errors at compile time if any enumerated type
        happens to conform to `RepeatableDecorator` -- `decorator_of` rejects
        that unconditionally (see below). There is no single call that is
        always safe to make blind; a generic enumerate-then-fetch loop should
        either know its decorator types are all non-repeatable, or branch on
        `conforms_to(tys[i], RepeatableDecorator)` and use `decorators_of`
        for the repeatable ones. Note that a repeatable type appears in this
        list once per occurrence, so such a loop visits it more than once and
        `decorators_of` returns every occurrence each time; dedupe the list
        first if occurrences must be counted once.

        Constraints:
            None. A non-struct `T` -- or a struct with no decorators -- yields
            an empty list rather than erroring, so a generic caller can ask
            without knowing what `T` is.

        Returns:
            A `TypeList` with one entry per decorator on the struct.

        Example:
            ```mojo
            @fieldwise_init
            struct tag(Decorator):
                pass

            @tag
            struct Thing:
                var x: Int

            def main():
                comptime tys = reflect[Thing].decorator_types()
                comptime assert tys[0] == tag
            ```
        """
        return {}

    @staticmethod
    def decorator_of[D: Movable]() -> Optional[D]:
        """Returns the `D` decorator applied to struct `T`, if present.

        Constraints:
            `T` must be a struct type. `D` must be `Movable` -- every
            decorator value already is, since it must be a legal comptime
            parameter value to have been recorded in the first place. `D`
            must not conform to `RepeatableDecorator`; use `decorators_of`
            for those.

        Parameters:
            D: The decorator type to look for.

        Returns:
            The decorator value, or `None` when `T` carries no `D`.

        Example:
            ```mojo
            struct serde(Decorator):
                var rename: StaticString

                def __init__(out self, rename: StaticString = ""):
                    self.rename = rename

            @serde(rename="NAME")
            struct Thing:
                var x: Int

            def main():
                comptime d = reflect[Thing].decorator_of[serde]()
                comptime assert Bool(d)
                comptime assert d.value().rename == "NAME"
            ```
        """
        comptime assert not conforms_to(D, RepeatableDecorator), String(
            t"decorator_of[D]() cannot be used with a decorator conforming"
            t" to RepeatableDecorator; a repeatable decorator may appear"
            t" more than once, so no spelling can safely return only one."
            t" Use decorators_of[D]() instead."
        )
        comptime found = _decorator_of[Self.T, D]()
        comptime if found.size == 0:
            return None
        else:
            return materialize[found[0]]()

    @staticmethod
    def decorators_of[D: Movable]() -> _decorators_of[Self.T, D]:
        """Returns every `D` decorator applied to struct `T`, nearest-first.

        For a decorator conforming to `RepeatableDecorator`, which may appear
        more than once on a single declaration. A non-repeatable decorator
        can appear at most once, so use `decorator_of[D]()` for those.

        Constraints:
            `T` must be a struct type. `D` must be `Movable` -- every
            decorator value already is, since it must be a legal comptime
            parameter value to have been recorded in the first place.

        Parameters:
            D: The decorator type to look for.

        Returns:
            A `ParameterList` with one entry per occurrence of `D` on the
            struct, nearest to the declaration first; empty when absent.

        Example:
            ```mojo
            struct alias_name(RepeatableDecorator):
                var wire: StaticString

                def __init__(out self, wire: StaticString):
                    self.wire = wire

            @alias_name("v")
            @alias_name("verbose")
            struct Flag:
                var on: Bool

            def main():
                comptime ds = reflect[Flag].decorators_of[alias_name]()
                comptime assert ds.size == 2
                comptime assert ds[0].wire == "verbose"
                comptime assert ds[1].wire == "v"
            ```
        """
        return {}

    @staticmethod
    def has_decorator[D: AnyType]() -> Bool:
        """Returns whether struct `T` carries a `D` decorator.

        Constraints:
            `T` must be a struct type.

        Parameters:
            D: The decorator type to look for.

        Returns:
            `True` when `T` carries a `D` decorator.
        """
        # Goes through the plural query, not `_decorator_of`: this must keep
        # working for a `D` conforming to `RepeatableDecorator`, which can
        # have more than one match -- `_decorator_of`'s folder now rejects
        # that (defense in depth for `decorator_of`'s at-most-one contract),
        # so routing a boolean "is there at least one" check through it would
        # hard-error instead of answering `True`.
        return _decorators_of[Self.T, D]().size > 0

    @staticmethod
    def field_names() -> Array[StaticString, _field_types_of[Self.T]().length]:
        """Returns the names of all fields in struct `T`.

        Constraints:
            `T` must be a struct type.

        Returns:
            An `Array` of `StaticString`, one entry per field.
        """
        comptime count = _field_types_of[Self.T]().length
        comptime raw = _field_names_of[Self.T]()

        # Safety: uninitialized=True is safe because the comptime for loop
        # below initializes every element.
        var result = Array[StaticString, count](uninitialized=True)

        comptime for i in range(raw.size):
            result[i] = comptime (StaticString(raw[i]))

        return result^

    @staticmethod
    def field_index[name: StringLiteral]() -> Int:
        """Returns the index of the field with the given name in struct `T`.

        Note: `T` must be a concrete type, not a generic type parameter, when
        looking up by name.

        Parameters:
            name: The name of the field to look up.

        Returns:
            The zero-based index of the field.
        """
        comptime str_value = name.value
        return Int(
            mlir_value=__mlir_attr[
                `#kgen.struct_field_index_by_name<`,
                Self.T,
                `, `,
                str_value,
                `> : index`,
            ]
        )

    # `field` is a parametric `comptime` member alias rather than a
    # static method, so callers spell `reflect[T].field["y"]` (no
    # parens) and get back `Reflected[FieldT]` directly. The result is
    # itself a reflection handle type — fully composable.
    comptime field[name: StringLiteral] = Reflected[
        __mlir_attr[
            `#kgen.struct_field_type_by_name<`,
            Self.T,
            `, `,
            name.value,
            `> : `,
            AnyType,
        ]
    ]
    """A reflection handle type for the named field's type.

    The result is `Reflected[FieldT]`, so `reflect[T].field["x"].T` can
    be used in type position and `.name()`, `.field_count()`, etc. compose
    directly without an additional `()`.

    Note: `T` must be a concrete type, not a generic type parameter, when
    looking up by name.

    Parameters:
        name: The name of the field.

    Example:
        ```mojo
        struct Point:
            var x: Int
            var y: Float64

        def main():
            comptime y_type = reflect[Point].field["y"]
            var v: y_type.T = 3.14  # y_type.T is Float64
        ```
    """

    # A separate name (not an overload of `field`) because `comptime`
    # member aliases cannot be overloaded on parameter type.
    comptime field_at[idx: Int] = Reflected[_field_types_of[Self.T]()[idx]]
    """A reflection handle type for the type of the field at the given index.

    The by-index dual of `field[name]`. Unlike the by-name form, it works
    when only the field index is available, such as inside a `comptime for` over
    a struct's fields, and `T` may be a generic type parameter. The result is
    `Reflected[FieldT]`, so `reflect[T].field_at[idx].T` reads as a type.

    Parameters:
        idx: The zero-based index of the field.

    Constraints:
        `T` must be a struct type. `idx` must be in range `[0, field_count())`.

    Example:
        ```mojo
        struct Point:
            var x: Int
            var y: Float64

        def main():
            comptime y_type = reflect[Point].field_at[1]
            var v: y_type.T = 3.14  # y_type.T is Float64
        ```
    """

    comptime member[name: StringLiteral] = Member[
        Self.T,
        Int(
            mlir_value=__mlir_attr[
                `#kgen.struct_field_index_by_name<`,
                Self.T,
                `, `,
                name.value,
                `> : index`,
            ]
        ),
    ]
    """A declaration handle on the named field.

    Distinct from `field[name]`, which is a handle on the field's *type*:
    `reflect[T].member["x"]` is a handle on the field declaration itself,
    including its decorators, while `reflect[T].field["x"]` is
    `Reflected[FieldT]` for the field's type directly. Reach the type from a
    member through `.type`.

    Note: `T` must be a concrete type, not a generic type parameter, when
    looking up by name.

    Parameters:
        name: The name of the field.
    """

    # A separate name (not an overload of `member`) because `comptime`
    # member aliases cannot be overloaded on parameter type -- the same
    # constraint that forces `field` and `field_at` apart.
    comptime member_at[idx: Int] = Member[Self.T, idx]
    """A declaration handle on the field at the given index.

    The by-index dual of `member[name]`. Unlike the by-name form it works
    when only the index is available, such as inside a `comptime for` over
    `field_count()`, and `T` may be a generic type parameter.

    Distinct from `field_at[idx]`, which is a handle on the field's *type*:
    `reflect[T].member_at[0]` is a handle on the field declaration, while
    `reflect[T].field_at[0]` is `Reflected[FieldT]` for its type directly.

    Parameters:
        idx: The zero-based index of the field.
    """

    # `nodebug` (not `builtin`) because the body emits a `lit.ref.struct.ger`
    # MLIR op, which is not legal inside a `@always_inline("builtin")` function.
    @staticmethod
    @always_inline("nodebug")
    def field_ref[
        idx: Int
    ](ref s: Self.T) -> ref[s] _field_types_of[Self.T]()[idx]:
        """Returns a reference to the field at the given index in `s`.

        Returns a reference rather than a copy, so this works with
        non-copyable field types and supports mutation through the result.

        Parameters:
            idx: The zero-based index of the field.

        Args:
            s: The struct value to access.

        Constraints:
            `T` must be a struct type. `idx` must be in range
            `[0, field_count())`.

        Returns:
            A reference to the field at the specified index, with the same
            mutability as `s`.

        Example:
            ```mojo
            @fieldwise_init
            struct Container:
                var id: Int
                var name: String

            def main():
                var c = Container(id=1, name="test")
                reflect[Container].field_ref[0](c) = 42  # mutates c.id
            ```
        """
        # Emit `lit.ref.struct.ger` with index-access form. The op accepts
        # StructType, ParamType, and ClosureType element types, so this works
        # for concrete structs, generic type parameters, and closures alike.
        return __get_litref_as_mvalue(
            __mlir_op.`lit.ref.struct.ger`[
                index=idx.__mlir_index__(),
                _type=__mlir_type[
                    `!lit.ref<`,
                    _field_types_of[Self.T]()[idx],
                    `, `,
                    origin_of(s)._mlir_origin,
                    `>`,
                ],
            ](__get_mvalue_as_litref(s))
        )

    @staticmethod
    def field_offset[
        *,
        name: StringLiteral,
        target: _TargetType = _current_target(),
    ]() -> Int:
        """Returns the byte offset of the named field within struct `T`.

        Accounts for alignment padding between fields. Computed using the
        target's data layout.

        Parameters:
            name: The name of the field.
            target: The target architecture (defaults to the current target).

        Constraints:
            `T` must be a struct type with a field of the given name.

        Returns:
            The byte offset of the field from the start of the struct.

        Example:
            ```mojo
            struct Point:
                var x: Int      # offset 0
                var y: Float64  # offset 8

            def main():
                comptime x_off = reflect[Point].field_offset[name="x"]()  # 0
                comptime y_off = reflect[Point].field_offset[name="y"]()  # 8
            ```
        """
        comptime str_value = name.value
        return Int(
            mlir_value=__mlir_attr[
                `#kgen.struct_field_offset_by_name<`,
                Self.T,
                `, `,
                str_value,
                `, `,
                target,
                `> : index`,
            ]
        )

    @staticmethod
    def field_offset[
        *,
        index: Int,
        target: _TargetType = _current_target(),
    ]() -> Int:
        """Returns the byte offset of the field at the given index.

        Accounts for alignment padding between fields. Computed using the
        target's data layout.

        Parameters:
            index: The zero-based index of the field.
            target: The target architecture (defaults to the current target).

        Constraints:
            `T` must be a struct type. `index` must be in range
            `[0, field_count())`.

        Returns:
            The byte offset of the field from the start of the struct.
        """
        return Int(
            mlir_value=__mlir_attr[
                `#kgen.struct_field_offset_by_index<`,
                Self.T,
                `, `,
                index.__mlir_index__(),
                `, `,
                target,
                `> : index`,
            ]
        )


struct Member[Owner: AnyType, idx: Int]:
    """A compile-time handle on a field *declaration* of struct `Owner`.

    Distinct from `Reflected[FieldT]`, which is a handle on the field's
    *type*: `reflect[T].member_at[0].decorator_of[serde]()` asks about the
    field declaration itself, while `reflect[T].field_at[0]` is the field's
    type. Reach the type from a member through `.type`.

    Obtain one from `reflect[T].member[name]` or `reflect[T].member_at[idx]`
    rather than constructing it directly.

    Parameters:
        Owner: The struct type the field belongs to.
        idx: The zero-based index of the field.

    Example:
        ```mojo
        struct serde(Decorator):
            var rename: StaticString

            def __init__(out self, rename: StaticString = ""):
                self.rename = rename

        struct Thing:
            @serde(rename="NAME")
            var name: String

        def main():
            comptime d = reflect[Thing].member_at[0].decorator_of[serde]()
            comptime assert Bool(d)
            comptime assert d.value().rename == "NAME"
        ```
    """

    comptime index = Self.idx
    """The zero-based index of this field within `Owner`."""

    comptime type = Reflected[_field_types_of[Self.Owner]()[Self.idx]]
    """A `Reflected` handle on this field's type."""

    @staticmethod
    def name() -> StaticString:
        """Returns the name of this field.

        Returns:
            The field's declared name.
        """
        return Reflected[Self.Owner].field_names()[Self.idx]

    @staticmethod
    def decorator_types() -> _field_decorator_types_of[Self.Owner, Self.idx]:
        """Returns the types of the decorators applied to this field.

        Entries are the decorator types, nearest to the declaration first.
        At most one decorator of a given type may appear on a declaration --
        unless that type conforms to `RepeatableDecorator`, in which case it
        may appear (and so be repeated in this list) more than once. An
        undecorated field yields an empty list.

        As with the struct-level form, an enumerate-then-fetch loop over
        these entries that calls `decorator_of[tys[i]]()` hard-errors at
        compile time if any entry conforms to `RepeatableDecorator` -- see
        the note on `Reflected.decorator_types`.

        Returns:
            A `TypeList` with one entry per decorator on the field.
        """
        return {}

    @staticmethod
    def decorator_of[D: Movable]() -> Optional[D]:
        """Returns the `D` decorator applied to this field, if present.

        Constraints:
            `D` must be `Movable` -- every decorator value already is, since
            it must be a legal comptime parameter value to have been recorded
            in the first place. `D` must not conform to `RepeatableDecorator`;
            use `decorators_of` for those.

        Parameters:
            D: The decorator type to look for.

        Returns:
            The decorator value, or `None` when the field carries no `D`.
        """
        comptime assert not conforms_to(D, RepeatableDecorator), String(
            t"decorator_of[D]() cannot be used with a decorator conforming"
            t" to RepeatableDecorator; a repeatable decorator may appear"
            t" more than once, so no spelling can safely return only one."
            t" Use decorators_of[D]() instead."
        )
        comptime found = _field_decorator_of[Self.Owner, Self.idx, D]()
        comptime if found.size == 0:
            return None
        else:
            return materialize[found[0]]()

    @staticmethod
    def decorators_of[
        D: Movable
    ]() -> _field_decorators_of[Self.Owner, Self.idx, D]:
        """Returns every `D` decorator applied to this field, nearest-first.

        For a decorator conforming to `RepeatableDecorator`, which may appear
        more than once on a single declaration. A non-repeatable decorator
        can appear at most once, so use `decorator_of[D]()` for those.

        Constraints:
            `D` must be `Movable` -- every decorator value already is, since
            it must be a legal comptime parameter value to have been recorded
            in the first place.

        Parameters:
            D: The decorator type to look for.

        Returns:
            A `ParameterList` with one entry per occurrence of `D` on the
            field, nearest to the declaration first; empty when absent.
        """
        return {}

    @staticmethod
    def has_decorator[D: AnyType]() -> Bool:
        """Returns whether this field carries a `D` decorator.

        Parameters:
            D: The decorator type to look for.

        Returns:
            `True` when the field carries a `D` decorator.
        """
        # See `Reflected.has_decorator`: goes through the plural query so
        # this keeps working for a repeatable `D` with more than one match.
        return _field_decorators_of[Self.Owner, Self.idx, D]().size > 0
