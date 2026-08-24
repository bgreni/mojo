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
# RUN: %parse-mojo-isolated %s --kgen-print-inline-type-values | FileCheck %s
# RUN: %parse-mojo-isolated %s | kgen-opt -lower-semantic-cf -check-lifetimes -lower-lit --kgen-print-inline-type-values | FileCheck %s --check-prefix=LOWER

# User-defined decorator values: a struct conforming to `Decorator`, applied
# with `@name` or `@name(args)`. Parenthesized operands are constructor
# arguments; the bare form gains an empty argument list, so both forms reduce
# to a constructor call whose value is stored in the `userDecorators`
# attribute. A decorator struct may not declare parameters (see
# `struct_decorator_types_errors.mojo`), so all payload is carried by fields.


@fieldwise_init
struct tag(Decorator):
    pass


struct serde(Decorator):
    var rename: StaticString
    var skip: Bool

    def __init__(out self, rename: StaticString = "", skip: Bool = False):
        self.rename = rename
        self.skip = skip


# Bare decorator: constructed with no arguments.
# CHECK-LABEL: lit.struct.decl @Tagged
# CHECK-SAME: userDecorators = #kgen<decorators[#kgen.param.expr<apply_result_slot, #kgen.param.expr<rebind, #kgen.symbol.constant<@{{.*}}::@tag::@"__init__()"> : {{.*}}> : {{.*}}> : !tag]>
@tag
struct Tagged(Movable where False):
    pass


# Parenthesized operands are constructor arguments (keyword form).
# CHECK-LABEL: lit.struct.decl @Renamed
# CHECK-SAME: userDecorators = #kgen<decorators[#kgen.param.expr<apply_result_slot, #kgen.param.expr<rebind, #kgen.symbol.constant<@{{.*}}::@serde::@"__init__(::StringSpan[False, ImmStaticOrigin, *()],::Bool)"> : {{.*}}"NAME"{{.*}}> : !serde]>
@serde(rename="NAME")
struct Renamed(Movable where False):
    pass


# Positional operands work too, and stacked decorators are all kept, nearest
# to the declaration first.
# CHECK-LABEL: lit.struct.decl @Stacked
# CHECK-SAME: userDecorators = #kgen<decorators[#kgen.param.expr<apply_result_slot, #kgen.param.expr<rebind, #kgen.symbol.constant<@{{.*}}::@serde::@"__init__(::StringSpan[False, ImmStaticOrigin, *()],::Bool)"> : {{.*}}"A"{{.*}}> : !serde, #kgen.param.expr<apply_result_slot, #kgen.param.expr<rebind, #kgen.symbol.constant<@{{.*}}::@tag::@"__init__()"> : {{.*}}> : {{.*}}> : !tag]>
@tag
@serde("A")
struct Stacked(Movable where False):
    pass


# A decorator value may coexist with a compiler (function) decorator; only
# the decorator value is forwarded to `userDecorators`.
def register(a: StringLiteral):
    return


# CHECK-LABEL: lit.struct.decl @Mixed
# CHECK-SAME: userDecorators = #kgen<decorators[#kgen.param.expr<apply_result_slot, #kgen.param.expr<rebind, #kgen.symbol.constant<@{{.*}}::@tag::@"__init__()"> : {{.*}}> : {{.*}}> : !tag]>
@tag
@register("hello")
struct Mixed(Movable where False):
    pass


# A decorator struct may be declared *after* the declaration that uses it:
# classification is a name lookup performed when body decorators are applied.
# CHECK-LABEL: lit.struct.decl @UsesLateTag
# CHECK-SAME: userDecorators = #kgen<decorators[#kgen.param.expr<apply_result_slot, #kgen.param.expr<rebind, #kgen.symbol.constant<@{{.*}}::@latetag::@"__init__()"> : {{.*}}> : {{.*}}> : !latetag]>
@latetag
struct UsesLateTag(Movable where False):
    pass


@fieldwise_init
struct latetag(Decorator):
    pass


# ===----------------------------------------------------------------------=== #
# Field decorators
# ===----------------------------------------------------------------------=== #


# CHECK-LABEL: lit.struct.decl @Thing
struct Thing(Movable where False):
    # CHECK: lit.struct.field name {decorators = #kgen<decorators[{{.*}}#kgen.symbol.constant<@{{.*}}::@serde::@"__init__{{.*}}"NAME"{{.*}}]>}
    @serde(rename="NAME")
    var name: Int

    # CHECK: lit.struct.field age {decorators = #kgen<decorators[{{.*}}#kgen.symbol.constant<@{{.*}}::@serde::@"__init__{{.*}}_mlir_value: scalar<bool> = true{{.*}}#kgen.symbol.constant<@{{.*}}::@tag::@"__init__()">{{.*}}: !tag]>}
    @tag
    @serde(skip=True)
    var age: Int

    # Undecorated fields carry no attribute.
    # CHECK: lit.struct.field plain :
    var plain: Int

    # Existing syntactic field decorators keep working alongside.
    # CHECK: lit.struct.field hidden {decorators = #kgen<decorators[{{.*}}#kgen.symbol.constant<@{{.*}}::@tag::@"__init__()">{{.*}}: !tag]>, isDocHidden}
    @tag
    @doc_hidden
    var hidden: Int


# Decorator arguments may refer to the enclosing struct's parameters via Self.
# CHECK-LABEL: lit.struct.decl @Parametric
struct Parametric[N: StringLiteral](Movable where False):
    # CHECK: lit.struct.field x {decorators = #kgen<decorators[{{.*}}#kgen.symbol.constant<@{{.*}}::@serde::@"__init__{{.*}}N.value{{.*}}]>}
    @serde(rename=Self.N)
    var x: Int


# The struct-level and field-level decorator namespaces are independent: a
# struct and one of its own fields may each carry a `@serde` without
# triggering the duplicate-decorator check.
# CHECK-LABEL: lit.struct.decl @SameDecoratorBothLevels
@serde(rename="outer")
struct SameDecoratorBothLevels(Movable where False):
    @serde(rename="inner")
    var x: Int


# ===----------------------------------------------------------------------=== #
# Repeatable decorators
# ===----------------------------------------------------------------------=== #

# A decorator conforming to `RepeatableDecorator` opts out of the "at most one
# decorator per struct" rule: the parser does not diagnose duplicates, and
# every occurrence is kept, nearest to the declaration first. This is
# per-struct-identity, unlike the (still-enforced) default -- see
# `struct_decorator_types_errors.mojo` for the non-repeatable case, which must
# keep erroring even when a repeatable decorator is also present.


struct label(RepeatableDecorator):
    var wire: StaticString

    def __init__(out self, wire: StaticString = ""):
        self.wire = wire


# CHECK-LABEL: lit.struct.decl @Repeated
# CHECK-SAME: userDecorators = #kgen<decorators[#kgen.param.expr<apply_result_slot, #kgen.param.expr<rebind, #kgen.symbol.constant<@{{.*}}::@label::@"__init__(::StringSpan[False, ImmStaticOrigin, *()])"> : {{.*}}"verbose"{{.*}}> : !label, #kgen.param.expr<apply_result_slot, #kgen.param.expr<rebind, #kgen.symbol.constant<@{{.*}}::@label::@"__init__(::StringSpan[False, ImmStaticOrigin, *()])"> : {{.*}}"v"{{.*}}> : !label]>
@label("v")
@label("verbose")
struct Repeated(Movable where False):
    pass


# ===----------------------------------------------------------------------=== #
# Site-derived parameter binding
# ===----------------------------------------------------------------------=== #

# A decorator struct may declare exactly one parameter (two or more is an
# error -- see `struct_decorator_types_errors.mojo`). The compiler binds it,
# never the user: to the field's type for a field decorator, and to the
# struct's own type for a struct decorator.
#
# The property this exists for is that the binding does not depend on which
# constructor arguments were written. `@tagged("a")` and a bare `@tagged` on
# two `Int` fields are the *same* type; under inference-from-arguments they
# would not be, and a reader querying one spelling would silently miss the
# other. The checks below assert that explicitly.


struct tagged[FieldT: AnyType](Decorator):
    var note: StaticString

    def __init__(out self, note: StaticString = ""):
        self.note = note


# CHECK-LABEL: lit.struct.decl @Bound
struct Bound(Movable where False):
    # CHECK: lit.struct.field x {decorators = #kgen<decorators[{{.*}}"a"{{.*}}: !lit.struct<#tagged <:!AnyType !Int>>]>}
    @tagged("a")
    var x: Int

    # Same `Int` binding as `x`, from the bare spelling with no arguments at
    # all -- this pairing is the whole point of deriving from the site.
    # CHECK: lit.struct.field y {decorators = #kgen<decorators[{{.*}}: !lit.struct<#tagged <:!AnyType !Int>>]>}
    @tagged
    var y: Int

    # A differently-typed field gets a different binding.
    # CHECK: lit.struct.field s {decorators = #kgen<decorators[{{.*}}: !lit.struct<#tagged <:!AnyType @{{.*}}StringSpan{{.*}}>>]>}
    @tagged("s")
    var s: StaticString


# A struct decorator binds to the struct's own type.
# CHECK-LABEL: lit.struct.decl @StructBound
# CHECK-SAME: userDecorators = #kgen<decorators[{{.*}}"outer"{{.*}}: !lit.struct<#tagged <:!AnyType !StructBound>>]>
@tagged("outer")
struct StructBound(Movable where False):
    var q: Int


# For a generic struct that is the *parameterized* self type, so the binding
# is rebound per instantiation like any other parameter reference.
# CHECK-LABEL: lit.struct.decl @GenericBound
# CHECK-SAME: userDecorators = #kgen<decorators[{{.*}}: !lit.struct<#tagged <:!AnyType @{{.*}}::@GenericBound<:!Int N>>>]>
@tagged("gen")
struct GenericBound[N: Int](Movable where False):
    var q: Int


# ===----------------------------------------------------------------------=== #
# LowerLIT forwards decorator values onto kgen.struct.generator
# ===----------------------------------------------------------------------=== #

# The decorator values survive lowering as KGEN constant values: a fully
# foldable value like `tag()` reduces to a plain (nameless) empty struct
# constant, while a value that embeds a runtime pointer (`serde`'s
# `StaticString` field) keeps a symbolic reference to the lowered constructor.
#
# Because the fully-foldable case reduces to an anonymous structural type
# (`!kgen.struct<() memoryOnly>`, no reference back to `tag`), two distinct
# fieldless decorator structs would otherwise be indistinguishable once
# lowered -- see `decoratorTypeNames` below, a parallel array (one entry per
# `decorators` entry, same order) that records which decorator struct
# produced each value and is therefore the only way to recover decorator
# identity post-lowering.
#
# The `attributes {...}` dictionary is printed after the generator body, so
# the checks below are plain (not -SAME) checks.
# LOWER-LABEL: kgen.struct.generator @"struct_decorator_types::Tagged"
# LOWER: attributes {decoratorTypeNames = [@"struct_decorator_types::tag"], decorators = #kgen<decorators[#kgen.struct<> : !kgen.struct<() memoryOnly>]>}

# Compiler (function) decorators are NOT forwarded; only decorator values are.
# LOWER-LABEL: kgen.struct.generator @"struct_decorator_types::Mixed"
# LOWER-NOT: register
# LOWER: attributes {decoratorTypeNames = [@"struct_decorator_types::tag"], decorators = #kgen<decorators[#kgen.struct<> : !kgen.struct<() memoryOnly>]>}

# Regression case for decorator identity: `UsesLateTag` is decorated with
# `latetag`, not `tag`. Its `decorators` value folds to the exact same
# byte-identical `#kgen.struct<> : !kgen.struct<() memoryOnly>` constant as
# `Tagged`'s and `Mixed`'s above -- only `decoratorTypeNames` distinguishes
# which decorator struct was actually applied.
# LOWER-LABEL: kgen.struct.generator @"struct_decorator_types::UsesLateTag"
# LOWER: attributes {decoratorTypeNames = [@"struct_decorator_types::latetag"], decorators = #kgen<decorators[#kgen.struct<> : !kgen.struct<() memoryOnly>]>}

# `Thing` has field decorators but no struct-level ones, so only
# `fieldDecorators`/`fieldDecoratorTypeNames` are set: one entry per field, in
# field order, with an empty entry for the undecorated `plain` field. `age`
# stacks `serde` (a `#kgen.struct<>`-embedding fold, hence needing its own
# name) with the fieldless `tag` (again indistinguishable from `latetag`
# without `fieldDecoratorTypeNames`).
# LOWER-LABEL: kgen.struct.generator @"struct_decorator_types::Thing"
# LOWER: attributes {fieldDecoratorTypeNames = [
# LOWER-SAME: [@"struct_decorator_types::serde"],
# LOWER-SAME: [@"struct_decorator_types::serde", @"struct_decorator_types::tag"],
# LOWER-SAME: [],
# LOWER-SAME: [@"struct_decorator_types::tag"]],
# LOWER-SAME: fieldDecorators = [
# LOWER-SAME: #kgen<decorators[{{.*}}"NAME"
# LOWER-SAME: #kgen<decorators[{{.*}}#kgen<simd true> : !kgen.scalar<bool>{{.*}}, #kgen.struct<> : !kgen.struct<() memoryOnly>]>,
# LOWER-SAME: #kgen<decorators[]>,
# LOWER-SAME: #kgen<decorators[#kgen.struct<> : !kgen.struct<() memoryOnly>]>]}

# `Repeated` carries the repeatable `label` decorator twice; both entries
# survive lowering, nearest to the declaration first, same as any other
# decorator. Only `decoratorTypeNames` needs to say `label` twice -- there is
# no separate "repeated" encoding, it is just two ordinary entries that happen
# to share an identity.
# LOWER-LABEL: kgen.struct.generator @"struct_decorator_types::Repeated"
# LOWER: attributes {decoratorTypeNames = [@"struct_decorator_types::label", @"struct_decorator_types::label"], decorators = #kgen<decorators[{{.*}}"verbose"{{.*}}, {{.*}}"v"{{.*}}]>}


# A site-derived parameter binding survives lowering the same way: identity is
# still just the base struct's symbol in `decoratorTypeNames`, and the binding
# rides along inside the decorator value (here, in the parameterization of the
# lowered constructor symbol). Nothing new is recorded for it, because the site
# reconstructs it -- which is why one parameter is affordable and two are not.
#
# `x` and `y` carry the *identical* `SIMD<index, 1>` binding despite being
# written `@tagged("a")` and bare `@tagged`; `s` carries `StringSpan`.
# LOWER-LABEL: kgen.struct.generator @"struct_decorator_types::Bound"
# LOWER: attributes {fieldDecoratorTypeNames = [
# LOWER-SAME: [@"struct_decorator_types::tagged"],
# LOWER-SAME: [@"struct_decorator_types::tagged"],
# LOWER-SAME: [@"struct_decorator_types::tagged"]],
# LOWER-SAME: fieldDecorators = [
# LOWER-SAME: <:type [typevalue<#kgen.genref<@"std::builtin::stubs::SIMD"<:dtype index, 1>>>, scalar<index>]>
# LOWER-SAME: "a"
# LOWER-SAME: <:type [typevalue<#kgen.genref<@"std::builtin::stubs::SIMD"<:dtype index, 1>>>, scalar<index>]>
# LOWER-SAME: <:type [typevalue<#kgen.genref<@"std::builtin::stubs::StringSpan"
# LOWER-SAME: "s"

# A struct decorator's binding is the struct's own lowered type.
# LOWER-LABEL: kgen.struct.generator @"struct_decorator_types::StructBound"
# LOWER: attributes {decoratorTypeNames = [@"struct_decorator_types::tagged"],
# LOWER-SAME: <:type [typevalue<#kgen.genref<@"struct_decorator_types::StructBound">>
# LOWER-SAME: "outer"

# And for a generic struct it is the parameterized self type, still carrying
# the parameter reference so it specializes per instantiation.
# LOWER-LABEL: kgen.struct.generator @"struct_decorator_types::GenericBound"
# LOWER: attributes {decoratorTypeNames = [@"struct_decorator_types::tagged"],
# LOWER-SAME: <:type [typevalue<#kgen.genref<@"struct_decorator_types::GenericBound"<:scalar<index> N>>>
# LOWER-SAME: "gen"
