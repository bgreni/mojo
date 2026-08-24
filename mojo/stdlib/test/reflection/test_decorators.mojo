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

from std.reflection import Decorator, Member, RepeatableDecorator, reflect
from std.testing import TestSuite, assert_equal, assert_false, assert_true
from test_utils.reflection import DecoratedFixture, FixtureName, FixtureTag


@fieldwise_init
struct tag(Decorator):
    pass


struct serde(Decorator):
    var rename: StaticString
    var skip: Bool

    def __init__(out self, rename: StaticString = "", skip: Bool = False):
        self.rename = rename
        self.skip = skip


@fieldwise_init
struct rename_all(Decorator):
    var policy: StaticString


@rename_all("camelCase")
struct Thing:
    @serde(rename="NAME")
    var name: String
    var plain: Int


@tag
@serde(rename="NAME")
struct Decorated:
    var x: Int


# Distinct decorator types stacked on one declaration -- the no-duplicates
# rule forbids two `@serde`s here, so `Stacked` exercises `serde` and
# `rename_all` together instead.
@serde(skip=True)
@rename_all("snake_case")
struct Stacked:
    var x: Int


struct Plain:
    var x: Int


# A second fieldless decorator. `tag()` and `marker()` lower to byte-identical
# constants, so `TagOnly` and `MarkerOnly` are only distinguishable by the
# decorator identity recorded alongside the value -- see
# `test_generic_decorator_types`.
@fieldwise_init
struct marker(Decorator):
    pass


@tag
struct TagOnly:
    var x: Int


@marker
struct MarkerOnly:
    var x: Int


# A dependent decorator payload: `Self.N` is only known once `Parametric` is
# specialized. This is the design's one live risk -- that a dependent value
# survives storage in a decorator attribute and retrieval through the folder.
# Struct-level and field-level decorators are independent namespaces, so the
# same dependent decorator appears on both the struct and its own field here,
# to exercise the risk through both the struct-level and field-level paths.
@serde(rename=Self.N)
struct Parametric[N: StaticString]:
    @serde(rename=Self.N)
    var x: Int


# Payload-legality finding (see task-5-report.md): the design anticipated
# that a decorator field of a type that is "not a legal comptime parameter
# value" -- `String` was the example given -- would need a decorator-specific
# diagnostic. In the current compiler `String` (and other heap-backed types,
# verified separately with `List[Int]`) *are* legal comptime parameter
# values: `tryEmitDecoratorValue` performs no additional legality check
# beyond `Decorator` conformance and the no-parameters rule, and the value
# below round-trips correctly through `decorator_of`. This test pins that
# behavior rather than papering over it.
struct string_payload(Decorator):
    var s: String

    def __init__(out self, s: String):
        self.s = s


@string_payload("hello")
struct StringPayload:
    var x: Int


# A decorator conforming to `RepeatableDecorator` opts out of the
# no-duplicates rule: the parser does not diagnose repeats, `decorators_of`
# returns every occurrence, and `decorator_of` is rejected at compile time.
struct alias_name(RepeatableDecorator):
    var wire: StaticString

    def __init__(out self, wire: StaticString = ""):
        self.wire = wire


@alias_name("v")
@alias_name("verbose")
struct Flag:
    var on: Bool


struct RepeatedField:
    @alias_name("a")
    @alias_name("b")
    var x: Int


# ===----------------------------------------------------------------------=== #
# Site-derived parameter binding
# ===----------------------------------------------------------------------=== #

# A decorator struct may declare exactly one parameter. The compiler binds it
# to the type of the declaration the decorator is attached to -- never to
# anything inferred from the constructor arguments. Reading it back therefore
# means naming that parameterization: `decorator_of[tagged[Int]]()`.


struct tagged[FieldT: AnyType](Decorator):
    var note: StaticString

    def __init__(out self, note: StaticString = ""):
        self.note = note


@tagged("outer")
struct Bound:
    @tagged("a")
    var x: Int

    @tagged("b")
    var y: String

    # Bare spelling, no constructor arguments at all: must still be
    # `tagged[Int]`, exactly like `x`.
    @tagged
    var z: Int


def _is_empty(s: String) -> Bool:
    return s.byte_length() == 0


def _never(s: String) -> Bool:
    return False


# The shape the whole gap exists for: a payload whose *own type* is written in
# terms of the site-derived parameter. `skip_if` is a function of the field's
# type and `default` is a value of it, neither of which can be spelled without
# the parameter. (`thin` is required -- the bare `def(T) -> R` spelling is a
# trait type and is rejected as a field type.)
#
# The bound on `FieldT` is checked at every site the decorator is attached to,
# whatever arguments were passed, so it must be no stronger than the payloads
# actually need. Storing a *value* of `FieldT` needs `Movable & Deinitable`:
# `Deinitable` for the field, `Movable` for the synthesized move constructor
# and for the `^` transfer below. Taking the argument `var` and transferring is
# what keeps it off `ImplicitlyCopyable` -- see `pred_dec` for why that
# matters.
struct field_dec[FieldT: Movable & Deinitable](Decorator):
    var rename: Optional[StaticString]
    var skip_if: Optional[def (Self.FieldT) thin -> Bool]
    var default: Optional[Self.FieldT]

    def __init__(
        out self,
        rename: Optional[StaticString] = None,
        skip_if: Optional[def (Self.FieldT) thin -> Bool] = None,
        var default: Optional[Self.FieldT] = None,
    ):
        self.rename = rename
        self.skip_if = skip_if
        self.default = default^


struct Serialized:
    @field_dec(
        rename=StaticString("n"),
        skip_if=_is_empty,
        default=String("unknown"),
    )
    var name: String

    @field_dec(rename=StaticString("c"), default=7)
    var count: Int


# The spec's own justifying example, verbatim in shape: two fields of the
# *same* type carrying different *argument subsets* of one parameterized
# decorator. Both are `field_dec[String]`, so one `decorator_of` spelling
# reads both. Under inference-from-arguments the first would be
# `field_dec[NoneType]` and only the second `field_dec[String]`.
struct SpecShape:
    @field_dec(rename=StaticString("n"))
    var a: String

    @field_dec(skip_if=_is_empty)
    var b: String


# A decorator that stores no *value* of `FieldT` needs nothing of it at all:
# `Optional` is conditionally conforming, and a function type over `FieldT`
# does not require `FieldT` to be movable or copyable. Keeping the bound at
# `AnyType` is not cosmetic -- the bound is enforced at the attachment site
# regardless of which arguments were passed, and `List`/`Dict` conform to
# `Copyable` but *not* `ImplicitlyCopyable`, so a decorator declared
# `[FieldT: ImplicitlyCopyable & ...]` cannot be attached to a collection
# field at all. `test_collection_field_decorator` pins that boundary.
struct pred_dec[FieldT: AnyType](Decorator):
    var rename: Optional[StaticString]
    var skip_if: Optional[def (Self.FieldT) thin -> Bool]

    def __init__(
        out self,
        rename: Optional[StaticString] = None,
        skip_if: Optional[def (Self.FieldT) thin -> Bool] = None,
    ):
        self.rename = rename
        self.skip_if = skip_if


struct HasCollections:
    @pred_dec(rename=StaticString("xs"))
    var xs: List[Int]

    @pred_dec(skip_if=_is_empty)
    var name: String

    # The combination the `Movable & Deinitable` bound exists to permit: a
    # decorator storing a *value* of the field's type, attached to a
    # collection field. `List` is unconditionally `Movable` and
    # `Deinitable where conforms_to(T, Deinitable)`, so it satisfies that
    # bound while failing the `ImplicitlyCopyable` one a plain
    # `self.default = default` would have forced.
    @field_dec(
        rename=StaticString("tags"),
        default=List[Int](1, 2, 3, __list_literal__=None),
    )
    var tags: List[Int]


# The one shape where a stored parameterization and a queried one are spelled
# in *different worlds*: the field's type is the enclosing struct's own
# parameter, so the decorator is stored as `tagged[T]` and only rebinding it
# through the struct's parameter values turns it into `tagged[Int]`. Every
# other fixture here declares concrete field types, for which the folder's
# rebinding step is an identity function and proves nothing.
#
# The struct-level decorator is the self-referential half of the same thing:
# `tagged[DependentBound[T]]`, rebuilt while evaluating an attribute *on*
# `DependentBound`.
@tagged("outer")
struct DependentBound[T: Movable & Deinitable]:
    @tagged("dep")
    var x: Self.T

    # A concrete field alongside, so the two paths are exercised on one site.
    @tagged("plain")
    var y: Int


# Gap 4 x gap 2: a repeatable decorator may be parameterized too. All its
# occurrences on one declaration necessarily share a binding, since the
# binding comes from the declaration -- which is also why keying the parser's
# duplicate check on the base struct alone stays exact.
struct rep_tagged[FieldT: AnyType](RepeatableDecorator):
    var wire: StaticString

    def __init__(out self, wire: StaticString = ""):
        self.wire = wire


struct RepeatedBound:
    @rep_tagged("a")
    @rep_tagged("b")
    var x: Int


def test_decorator_trait() raises:
    assert_true(conforms_to(tag, Decorator))
    assert_false(conforms_to(Int, Decorator))


def test_decorator_types() raises:
    comptime tys = reflect[Decorated].decorator_types()
    assert_equal(tys.length, 2)
    comptime assert tys[0] == serde
    comptime assert tys[1] == tag
    assert_equal(reflect[Plain].decorator_types().length, 0)
    assert_equal(reflect[Int].decorator_types().length, 0)


def test_decorator_of_present() raises:
    comptime d = reflect[Decorated].decorator_of[serde]()
    comptime assert Bool(d)
    comptime assert d.value().rename == "NAME"
    comptime assert not d.value().skip


def test_decorator_of_absent() raises:
    comptime assert not Bool(reflect[Plain].decorator_of[serde]())

    # `Thing` carries `@serde` on its `name` field, not on the struct itself,
    # so a struct-level query for `serde` finds nothing.
    comptime assert not Bool(reflect[Thing].decorator_of[serde]())


def test_has_decorator() raises:
    assert_true(reflect[Decorated].has_decorator[serde]())
    assert_true(reflect[Decorated].has_decorator[tag]())
    assert_false(reflect[Plain].has_decorator[tag]())


def test_stacked_decorators() raises:
    comptime s = reflect[Stacked].decorator_of[serde]()
    comptime assert Bool(s)
    comptime assert s.value().skip

    comptime p = reflect[Stacked].decorator_of[rename_all]()
    comptime assert Bool(p)
    comptime assert p.value().policy == "snake_case"


def _rename[T: AnyType]() -> StaticString:
    """Exercises the elaboration-time path: `T` is generic here."""
    comptime d = reflect[T].decorator_of[serde]()
    comptime if d:
        return materialize[d.value().rename]()
    else:
        return "none"


def test_generic_decorators() raises:
    assert_equal(String(_rename[Decorated]()), "NAME")
    assert_equal(String(_rename[Plain]()), "none")


def test_dependent_decorator_value() raises:
    """Struct-level: the gate on dependent values surviving the folder's
    evaluation."""
    comptime d = reflect[Parametric["foo"]].decorator_of[serde]()
    comptime assert Bool(d)
    comptime assert d.value().rename == "foo"


def test_dependent_decorator_value_field() raises:
    """Field-level equivalent of `test_dependent_decorator_value`: the only
    end-to-end proof that a dependent decorator payload survives
    specialization and reads back correctly through the field path (as
    opposed to the parser-suite elaboration tests, which assert counts only,
    since that suite's `StaticString` stubs are no-ops)."""
    comptime d = reflect[Parametric["foo"]].member_at[0].decorator_of[serde]()
    comptime assert Bool(d)
    comptime assert d.value().rename == "foo"


def test_member_decorators() raises:
    comptime d = reflect[Thing].member_at[0].decorator_of[serde]()
    comptime assert Bool(d)
    comptime assert d.value().rename == "NAME"

    # An undecorated field, and a struct whose decorators are all
    # struct-level.
    comptime assert not Bool(reflect[Thing].member_at[1].decorator_of[serde]())
    comptime assert not Bool(
        reflect[Decorated].member_at[0].decorator_of[serde]()
    )


def test_member_by_name() raises:
    comptime d = reflect[Thing].member["name"].decorator_of[serde]()
    comptime assert Bool(d)
    comptime assert d.value().rename == "NAME"


def test_member_basics() raises:
    assert_equal(String(reflect[Thing].member_at[0].name()), "name")
    assert_equal(reflect[Thing].member_at[1].index, 1)
    comptime assert reflect[Thing].member_at[1].type.T == Int
    assert_equal(reflect[Thing].member_at[0].decorator_types().length, 1)
    assert_equal(reflect[Thing].member_at[1].decorator_types().length, 0)
    assert_true(reflect[Thing].member_at[0].has_decorator[serde]())
    assert_false(reflect[Thing].member_at[1].has_decorator[serde]())


def _field_rename[T: AnyType, i: Int]() -> StaticString:
    """Exercises the elaboration-time path: `T` is generic here."""
    comptime d = reflect[T].member_at[i].decorator_of[serde]()
    comptime if d:
        return materialize[d.value().rename]()
    else:
        return "none"


def test_generic_member_decorators() raises:
    assert_equal(String(_field_rename[Thing, 0]()), "NAME")
    assert_equal(String(_field_rename[Thing, 1]()), "none")


def _decorator_type_count[T: AnyType]() -> Int:
    """Exercises the elaboration-time path: `T` is generic here, so
    `decorator_types()` is folded against the lowered struct generator rather
    than at parse time."""
    return reflect[T].decorator_types().length


def _nth_decorator_is[T: AnyType, i: Int, D: AnyType]() -> Bool:
    """Elaboration-time `decorator_types()`, compared against a named type."""
    comptime tys = reflect[T].decorator_types()
    comptime if i >= tys.length:
        return False
    else:
        return tys[i] == D


def _same_first_decorator[T: AnyType, U: AnyType]() -> Bool:
    """Elaboration-time `decorator_types()`, compared against itself.

    Lowering anonymizes a decorator value's type -- every fieldless decorator
    becomes the same empty struct type -- so this is the sharpest check that
    the enumeration reports identity rather than layout: it names no decorator,
    and so cannot pass by both sides being equally anonymous."""
    comptime a = reflect[T].decorator_types()
    comptime b = reflect[U].decorator_types()
    comptime if a.length == 0 or b.length == 0:
        return False
    else:
        return a[0] == b[0]


def test_generic_decorator_types() raises:
    assert_equal(_decorator_type_count[Decorated](), 2)
    assert_equal(_decorator_type_count[Plain](), 0)

    assert_true(_nth_decorator_is[Decorated, 0, serde]())
    assert_true(_nth_decorator_is[Decorated, 1, tag]())
    assert_false(_nth_decorator_is[Decorated, 1, marker]())

    assert_true(_nth_decorator_is[TagOnly, 0, tag]())
    assert_false(_nth_decorator_is[TagOnly, 0, marker]())
    assert_true(_same_first_decorator[TagOnly, TagOnly]())
    assert_false(_same_first_decorator[TagOnly, MarkerOnly]())


def test_enumerate_then_fetch() raises:
    """Every decorator on a declaration, read without naming its type up front.
    """
    comptime tys = reflect[Decorated].decorator_types()
    comptime assert tys.length == 2

    var seen = 0
    comptime for i in range(tys.length):
        comptime d = reflect[Decorated].decorator_of[tys[i]]()
        comptime assert Bool(d)
        seen += 1
    assert_equal(seen, 2)


def test_enumerate_then_fetch_field() raises:
    comptime tys = reflect[Thing].member_at[0].decorator_types()
    comptime assert tys.length == 1
    comptime for i in range(tys.length):
        comptime d = reflect[Thing].member_at[0].decorator_of[tys[i]]()
        comptime assert Bool(d)


def _enumerate_count[T: AnyType]() -> Int:
    """Elaboration-time enumerate-then-fetch."""
    comptime tys = reflect[T].decorator_types()
    var n = 0
    comptime for i in range(tys.length):
        comptime if Bool(reflect[T].decorator_of[tys[i]]()):
            n += 1
    return n


def test_enumerate_then_fetch_generic() raises:
    assert_equal(_enumerate_count[Decorated](), 2)
    assert_equal(_enumerate_count[Plain](), 0)


def _field_enumerate_count[T: AnyType, i: Int]() -> Int:
    """Field-level, elaboration-time enumerate-then-fetch -- the missing
    combination flagged in Task 2's review: struct x field, times concrete x
    generic T, is four combinations, and only this one (field + generic) was
    absent. This is exactly the coverage shape that let a Phase 1 bug hide
    for four tasks, per the ledger."""
    comptime tys = reflect[T].member_at[i].decorator_types()
    var n = 0
    comptime for j in range(tys.length):
        comptime if Bool(reflect[T].member_at[i].decorator_of[tys[j]]()):
            n += 1
    return n


def test_enumerate_then_fetch_field_generic() raises:
    assert_equal(_field_enumerate_count[Thing, 0](), 1)
    assert_equal(_field_enumerate_count[Thing, 1](), 0)


def _field_decorator_type_count[T: AnyType, i: Int]() -> Int:
    """The field-level enumeration, on the elaboration-time path."""
    return reflect[T].member_at[i].decorator_types().length


def _nth_field_decorator_is[T: AnyType, i: Int, j: Int, D: AnyType]() -> Bool:
    comptime tys = reflect[T].member_at[i].decorator_types()
    comptime if j >= tys.length:
        return False
    else:
        return tys[j] == D


def test_generic_member_decorator_types() raises:
    assert_equal(_field_decorator_type_count[Thing, 0](), 1)
    assert_equal(_field_decorator_type_count[Thing, 1](), 0)
    assert_true(_nth_field_decorator_is[Thing, 0, 0, serde]())
    assert_false(_nth_field_decorator_is[Thing, 0, 0, rename_all]())


def test_member_is_exported() raises:
    """`Member` is re-exported from `std.reflection`, so a caller can name the
    handle type directly instead of only reaching it through
    `reflect[T].member_at[i]`."""
    comptime m = Member[Thing, 0]
    assert_equal(String(m.name()), "name")
    assert_equal(m.index, 0)
    comptime d = m.decorator_of[serde]()
    comptime assert Bool(d)
    comptime assert d.value().rename == "NAME"


def test_string_payload_is_accepted() raises:
    """A `String` decorator field is accepted, not rejected -- see the
    payload-legality finding above and in task-5-report.md."""
    assert_true(reflect[StringPayload].has_decorator[string_payload]())
    comptime d = reflect[StringPayload].decorator_of[string_payload]()
    comptime assert Bool(d)
    assert_equal(String(materialize[d.value().s]()), "hello")


def test_cross_module_decorators() raises:
    comptime tys = reflect[DecoratedFixture].decorator_types()
    assert_equal(tys.length, 2)
    comptime assert tys[0] == FixtureName
    comptime assert tys[1] == FixtureTag

    comptime n = reflect[DecoratedFixture].decorator_of[FixtureName]()
    comptime assert Bool(n)
    comptime assert n.value().wire == "wire_a"

    assert_true(reflect[DecoratedFixture].has_decorator[FixtureTag]())

    # Field-level, cross-module: `DecoratedFixture.a` carries its own
    # `@FixtureName`, independent of the struct-level one above.
    comptime f = reflect[DecoratedFixture].member_at[0].decorator_of[
        FixtureName
    ]()
    comptime assert Bool(f)
    comptime assert f.value().wire == "field_a"


def test_repeatable_decorator_trait() raises:
    assert_true(conforms_to(alias_name, RepeatableDecorator))
    assert_true(conforms_to(alias_name, Decorator))
    assert_false(conforms_to(serde, RepeatableDecorator))


def test_repeatable_decorator() raises:
    # Nearest to the declaration first: `@alias_name("verbose")` is written
    # directly above `struct Flag`, so it is entry 0 -- same convention as
    # `decorator_types()` (see `test_decorator_types` above, where `serde`,
    # written closest to `struct Decorated`, is `tys[0]`).
    comptime ds = reflect[Flag].decorators_of[alias_name]()
    assert_equal(ds.size, 2)
    comptime assert ds[0].wire == "verbose"
    comptime assert ds[1].wire == "v"


def test_repeatable_decorator_absent() raises:
    assert_equal(reflect[Plain].decorators_of[alias_name]().size, 0)


def test_repeatable_decorator_types_enumerates_duplicates() raises:
    """`decorator_types()` has no dedup logic of its own -- a repeatable
    decorator applied twice shows up twice, nearest to the declaration
    first."""
    comptime tys = reflect[Flag].decorator_types()
    assert_equal(tys.length, 2)
    comptime assert tys[0] == alias_name
    comptime assert tys[1] == alias_name


def test_repeatable_decorator_has_decorator() raises:
    assert_true(reflect[Flag].has_decorator[alias_name]())
    assert_false(reflect[Plain].has_decorator[alias_name]())


def test_repeatable_field_decorator() raises:
    # `@alias_name("b")` is nearest to `var x: Int`, so it is entry 0 -- see
    # the ordering note in `test_repeatable_decorator` above.
    comptime ds = reflect[RepeatedField].member_at[0].decorators_of[
        alias_name
    ]()
    assert_equal(ds.size, 2)
    comptime assert ds[0].wire == "b"
    comptime assert ds[1].wire == "a"


def _repeatable_count[T: AnyType]() -> Int:
    """Elaboration-time `decorators_of`: `T` is generic here."""
    return reflect[T].decorators_of[alias_name]().size


def test_repeatable_decorator_generic() raises:
    assert_equal(_repeatable_count[Flag](), 2)
    assert_equal(_repeatable_count[Plain](), 0)


def _repeatable_field_count[T: AnyType, i: Int]() -> Int:
    return reflect[T].member_at[i].decorators_of[alias_name]().size


def test_repeatable_field_decorator_generic() raises:
    assert_equal(_repeatable_field_count[RepeatedField, 0](), 2)


def test_non_repeatable_still_singular() raises:
    """The default contract is unchanged for everything that does not opt in:
    at most one match, retrieved with `decorator_of`. This must not regress --
    it is the guarantee every existing `decorator_of[D]()` caller relies on."""
    comptime d = reflect[Decorated].decorator_of[serde]()
    comptime assert Bool(d)
    assert_equal(reflect[Flag].decorators_of[alias_name]().size, 2)


def test_site_derived_binding() raises:
    comptime a = reflect[Bound].member_at[0].decorator_of[tagged[Int]]()
    comptime assert Bool(a)
    comptime assert a.value().note == "a"

    comptime b = reflect[Bound].member_at[1].decorator_of[tagged[String]]()
    comptime assert Bool(b)
    comptime assert b.value().note == "b"

    # Wrong parameterization finds nothing, even though the base struct --
    # all that decorator identity records -- is the same `tagged`.
    comptime assert not Bool(
        reflect[Bound].member_at[0].decorator_of[tagged[String]]()
    )
    comptime assert not Bool(
        reflect[Bound].member_at[1].decorator_of[tagged[Int]]()
    )


def test_binding_is_argument_independent() raises:
    """`@tagged("a")` and a bare `@tagged` on same-typed fields agree.

    This is the property the design exists for. Under inference from the
    constructor arguments these two would be different types -- the bare form
    passes nothing to infer from -- and a reader querying one spelling would
    silently miss the other.
    """
    comptime t0 = reflect[Bound].member_at[0].decorator_of[tagged[Int]]()
    comptime t2 = reflect[Bound].member_at[2].decorator_of[tagged[Int]]()
    comptime assert Bool(t0)
    comptime assert Bool(t2)
    comptime assert t0.value().note == "a"
    comptime assert t2.value().note == ""


def test_struct_level_site_derived_binding() raises:
    """A struct decorator binds to the struct's own type."""
    comptime d = reflect[Bound].decorator_of[tagged[Bound]]()
    comptime assert Bool(d)
    comptime assert d.value().note == "outer"

    comptime assert not Bool(reflect[Bound].decorator_of[tagged[Int]]())


def test_site_derived_decorator_types() raises:
    """`decorator_types()` reports the *bound* type, so it round-trips back
    into `decorator_of` -- the gap-5 enumerate-then-fetch path, on a
    parameterized decorator."""
    comptime tys = reflect[Bound].member_at[0].decorator_types()
    assert_equal(tys.length, 1)
    comptime assert tys[0] == tagged[Int]
    comptime assert tys[0] != tagged[String]

    comptime for i in range(tys.length):
        comptime d = reflect[Bound].member_at[0].decorator_of[tys[i]]()
        comptime assert Bool(d)


def _bound_note[T: AnyType, i: Int]() -> StaticString:
    """Elaboration-time path: `T` is generic, so the query folds later."""
    comptime d = reflect[T].member_at[i].decorator_of[tagged[Int]]()
    comptime if d:
        return materialize[d.value().note]()
    else:
        return "none"


def test_site_derived_binding_generic() raises:
    assert_equal(String(_bound_note[Bound, 0]()), "a")
    assert_equal(String(_bound_note[Bound, 2]()), "")
    # Field 1 is a `String`, so a `tagged[Int]` query must not answer for it.
    assert_equal(String(_bound_note[Bound, 1]()), "none")


def test_function_value_payload() raises:
    """The `skip_if` shape: a payload whose type mentions the site-derived
    parameter, retrieved and invoked both at comptime and after
    `materialize`."""
    comptime d = reflect[Serialized].member_at[0].decorator_of[
        field_dec[String]
    ]()
    comptime assert Bool(d)
    comptime assert d.value().rename.value() == "n"
    comptime assert Bool(d.value().skip_if)
    # Called at compile time, on the retrieved value.
    comptime assert d.value().skip_if.value()("")
    comptime assert not d.value().skip_if.value()("x")
    # And crossing to runtime.
    var pred = materialize[d.value().skip_if.value()]()
    assert_true(pred(String("")))
    assert_false(pred(String("x")))


def test_default_value_payload() raises:
    """`default` is a *value* of the field's type, which is only spellable
    because the parameter is bound to that type."""
    comptime d = reflect[Serialized].member_at[0].decorator_of[
        field_dec[String]
    ]()
    comptime assert Bool(d)
    assert_equal(materialize[d.value().default.value()](), String("unknown"))

    comptime c = reflect[Serialized].member_at[1].decorator_of[
        field_dec[Int]
    ]()
    comptime assert Bool(c)
    comptime assert c.value().default.value() == 7
    # The `Int` field never got a predicate.
    comptime assert not Bool(c.value().skip_if)

    # And the two fields' decorators are genuinely different types: the
    # `Int` field does not answer a `field_dec[String]` query.
    comptime assert not Bool(
        reflect[Serialized].member_at[1].decorator_of[field_dec[String]]()
    )


def test_parameterized_has_decorator() raises:
    assert_true(reflect[Bound].member_at[0].has_decorator[tagged[Int]]())
    assert_false(reflect[Bound].member_at[0].has_decorator[tagged[String]]())
    assert_true(reflect[Bound].has_decorator[tagged[Bound]]())


def test_repeatable_parameterized_decorator() raises:
    # Nearest to the declaration first, same as any other repeatable
    # decorator: `@rep_tagged("b")` is written directly above `var x`.
    comptime ds = reflect[RepeatedBound].member_at[0].decorators_of[
        rep_tagged[Int]
    ]()
    assert_equal(ds.size, 2)
    comptime assert ds[0].wire == "b"
    comptime assert ds[1].wire == "a"

    # The parameterization filter applies to the plural query too.
    assert_equal(
        reflect[RepeatedBound]
        .member_at[0]
        .decorators_of[rep_tagged[String]]()
        .size,
        0,
    )


def test_spec_shape_argument_subsets() raises:
    """The spec's justifying case: two same-typed fields carrying different
    *argument subsets* of one parameterized decorator are the same type."""
    comptime a = reflect[SpecShape].member_at[0].decorator_of[
        field_dec[String]
    ]()
    comptime b = reflect[SpecShape].member_at[1].decorator_of[
        field_dec[String]
    ]()
    comptime assert Bool(a)
    comptime assert Bool(b)

    # One spelling, both fields -- which is the whole point.
    comptime assert a.value().rename.value() == "n"
    comptime assert not Bool(a.value().skip_if)
    comptime assert not Bool(b.value().rename)
    comptime assert Bool(b.value().skip_if)
    comptime assert b.value().skip_if.value()("")


def test_collection_value_payload() raises:
    """A `List[Int]` stored as a decorator payload, on a `List[Int]` field.

    This is the shape a serde library's `default=` lands on, and it is the
    one the `Movable & Deinitable` bound was narrowed to permit -- the
    decorator would not attach to this field at all under
    `ImplicitlyCopyable & Deinitable`, whatever arguments were passed.
    Retrieved and read, not merely declared.
    """
    comptime d = reflect[HasCollections].member_at[2].decorator_of[
        field_dec[List[Int]]
    ]()
    comptime assert Bool(d)
    comptime assert d.value().rename.value() == "tags"
    comptime assert Bool(d.value().default)
    comptime assert len(d.value().default.value()) == 3
    comptime assert d.value().default.value()[0] == 1

    # And across the comptime-to-runtime boundary, contents intact.
    var default = materialize[d.value().default.value()]()
    assert_equal(len(default), 3)
    assert_equal(default[0], 1)
    assert_equal(default[1], 2)
    assert_equal(default[2], 3)

    # A `List[Int]` field and a `String` field carry genuinely different
    # decorator types, same as any other pair.
    comptime assert not Bool(
        reflect[HasCollections].member_at[2].decorator_of[field_dec[String]]()
    )


def test_collection_field_decorator() raises:
    """A decorator whose parameter is bound `AnyType` attaches to a `List`
    field.

    The bound is checked at the attachment site whatever arguments were
    passed, and `List` is `Copyable` but not `ImplicitlyCopyable`, so a
    decorator that demanded `ImplicitlyCopyable` could not be attached here at
    all -- not even to set `rename`. Keeping `pred_dec`'s bound minimal is
    what makes this legal; see the negative half in
    `KGEN/test/mojo-integration/decorator_parameter_errors.mojo`.
    """
    comptime d = reflect[HasCollections].member_at[0].decorator_of[
        pred_dec[List[Int]]
    ]()
    comptime assert Bool(d)
    comptime assert d.value().rename.value() == "xs"

    comptime s = reflect[HasCollections].member_at[1].decorator_of[
        pred_dec[String]
    ]()
    comptime assert Bool(s)
    comptime assert s.value().skip_if.value()("")

    # Still two different types, exactly as for any other pair of field types.
    comptime assert not Bool(
        reflect[HasCollections].member_at[0].decorator_of[pred_dec[String]]()
    )


# Both kinds on one declaration, for the blind enumerate-then-fetch loop
# `decorator_types()` documents as the escape hatch.
@serde(rename="mix")
@alias_name("m1")
@alias_name("m2")
struct MixedDecorators:
    var x: Int


def _blind_decorator_count[T: AnyType]() -> Int:
    """The exact pattern `decorator_types()`'s docstring prescribes.

    A blind loop cannot call `decorator_of[tys[i]]()` unconditionally --
    that hard-errors for a `RepeatableDecorator` -- so it branches on
    `conforms_to` and uses the plural form for those. This exists to prove
    the documented workaround compiles; a documented escape hatch that does
    not compile is worse than none.
    """
    comptime tys = reflect[T].decorator_types()
    var n = 0
    comptime for i in range(tys.length):
        comptime if conforms_to(tys[i], RepeatableDecorator):
            n += reflect[T].decorators_of[tys[i]]().size
        else:
            comptime if Bool(reflect[T].decorator_of[tys[i]]()):
                n += 1
    return n


def test_blind_enumerate_then_fetch() raises:
    # `MixedDecorators` enumerates three entries -- `alias_name` twice, since
    # a repeatable type appears once per occurrence, plus `serde`. The
    # repeatable branch therefore runs twice and reports 2 each time, so a
    # naive blind loop counts 2 + 2 + 1. Worth knowing before writing one:
    # dedupe the list first if occurrences must be counted once.
    assert_equal(_blind_decorator_count[MixedDecorators](), 5)

    # Non-repeatable only, and undecorated, both take the other branch.
    assert_equal(_blind_decorator_count[Decorated](), 2)
    assert_equal(_blind_decorator_count[Plain](), 0)


def test_dependent_site_type_binding() raises:
    """The site type is the struct's own parameter, so it must be rebound.

    `DependentBound.x` stores `tagged[T]`; the query names `tagged[Int]`. The
    two only meet if the folder rebinds the stored site type through the
    struct's parameter values before comparing. For every other fixture in
    this file that rebinding is an identity function.
    """
    comptime a = reflect[DependentBound[Int]].member_at[0].decorator_of[
        tagged[Int]
    ]()
    comptime assert Bool(a)
    comptime assert a.value().note == "dep"

    # The same declaration, a different instantiation, a different type.
    comptime b = reflect[DependentBound[String]].member_at[0].decorator_of[
        tagged[String]
    ]()
    comptime assert Bool(b)
    comptime assert b.value().note == "dep"

    # And they do not answer each other's queries.
    comptime assert not Bool(
        reflect[DependentBound[Int]].member_at[0].decorator_of[tagged[String]]()
    )
    comptime assert not Bool(
        reflect[DependentBound[String]].member_at[0].decorator_of[tagged[Int]]()
    )

    # The concrete field next to it is unaffected by the instantiation.
    comptime c = reflect[DependentBound[String]].member_at[1].decorator_of[
        tagged[Int]
    ]()
    comptime assert Bool(c)
    comptime assert c.value().note == "plain"


def test_dependent_site_type_decorator_types() raises:
    """`decorator_types()` rebuilds the reported type from the recorded
    symbol plus the site, so it has to rebind the site type too."""
    comptime tys = reflect[DependentBound[Int]].member_at[0].decorator_types()
    assert_equal(tys.length, 1)
    comptime assert tys[0] == tagged[Int]
    comptime assert tys[0] != tagged[String]

    comptime stys = reflect[DependentBound[String]].member_at[
        0
    ].decorator_types()
    comptime assert stys[0] == tagged[String]

    # Enumerate then fetch, on the rebinding path.
    comptime for i in range(tys.length):
        comptime d = reflect[DependentBound[Int]].member_at[0].decorator_of[
            tys[i]
        ]()
        comptime assert Bool(d)


def test_struct_level_decorator_types() raises:
    """Struct-level `decorator_types()` on a parameterized decorator.

    The reported type is rebuilt from the recorded symbol with the site's own
    type bound back in -- self-referential by construction, since the site
    *is* the struct being reflected. Concrete struct first, then the generic
    one, where the binding additionally has to be rebound per instantiation.
    """
    comptime tys = reflect[Bound].decorator_types()
    assert_equal(tys.length, 1)
    comptime assert tys[0] == tagged[Bound]
    comptime assert tys[0] != tagged[Int]

    comptime d = reflect[Bound].decorator_of[tys[0]]()
    comptime assert Bool(d)
    comptime assert d.value().note == "outer"

    comptime gtys = reflect[DependentBound[Int]].decorator_types()
    assert_equal(gtys.length, 1)
    comptime assert gtys[0] == tagged[DependentBound[Int]]
    comptime assert gtys[0] != tagged[DependentBound[String]]

    comptime g = reflect[DependentBound[Int]].decorator_of[
        tagged[DependentBound[Int]]
    ]()
    comptime assert Bool(g)
    comptime assert g.value().note == "outer"


def _dependent_note[T: Movable & Deinitable, Q: AnyType]() -> StaticString:
    """Elaboration-time rebinding: both the site and the query are symbolic
    until `DependentBound[T]` is concretized."""
    comptime d = reflect[DependentBound[T]].member_at[0].decorator_of[
        tagged[Q]
    ]()
    comptime if d:
        return materialize[d.value().note]()
    else:
        return "none"


def test_dependent_site_type_binding_generic() raises:
    assert_equal(String(_dependent_note[Int, Int]()), "dep")
    assert_equal(String(_dependent_note[String, String]()), "dep")
    # Mismatched: the site rebinds to `String`, the query asks for `Int`.
    assert_equal(String(_dependent_note[String, Int]()), "none")
    assert_equal(String(_dependent_note[Int, String]()), "none")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
