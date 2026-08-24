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
# RUN: %parse-mojo-isolated %s | FileCheck %s
# RUN: %parse-mojo-isolated %s | kgen-opt -lower-semantic-cf -check-lifetimes -lower-lit -elaborate-generators="use-parametric-interpret=false" | FileCheck %s --check-prefix=ELAB
# RUN: %parse-mojo-isolated %s | kgen-opt -lower-semantic-cf -check-lifetimes -lower-lit -elaborate-generators="use-parametric-interpret=true" | FileCheck %s --check-prefix=ELAB

# The decorator reflection attributes exercised here: `*_decorator_types`
# enumerates the *types* of the decorator values applied to a declaration,
# struct- and field-level; `*_decorator_of` selects the single value of a
# requested decorator type, as a param_list of size 0 or 1. That 0-or-1 shape
# held unconditionally when this comment was written, because the parser
# forbade two decorators sharing a base struct on one declaration. Since
# `RepeatableDecorator` (see `decls/decorators/struct_decorator_types.mojo`),
# a repeatable decorator may legitimately have more than one match, so
# `*_decorator_of` is no longer capped by the parser alone -- it caps itself
# (`requireAtMostOneMatch` in `KGENAttrsFolders.cpp`) and rejects a
# repeatable decorator's second match rather than silently returning the
# first. This file only covers the non-repeatable path; the plural
# `*_decorators_of` attributes that return every match are not exercised
# here -- see `decorator_repeatable_errors.mojo` for the repeatable-specific
# coverage, including this rejection.

# ===----------------------------------------------------------------------=== #
# Helpers
# ===----------------------------------------------------------------------=== #

comptime decorator_types[T: AnyType] = TypeList[
    __mlir_attr[
        `#kgen.struct_decorator_types<`,
        T,
        `> : !kgen.param_list<`,
        AnyType,
        `>`,
    ]
]

comptime decorator_of[T: AnyType, D: AnyType] = ParameterList[
    type=D,
    __mlir_attr[
        `#kgen.struct_decorator_of<`,
        T,
        `, `,
        D,
        `> : !kgen.param_list<`,
        +D,
        `>`,
    ],
]

comptime field_decorator_types[T: AnyType, idx: Int] = TypeList[
    __mlir_attr[
        `#kgen.struct_field_decorator_types<`,
        T,
        `, `,
        idx.__mlir_index__(),
        `> : !kgen.param_list<`,
        AnyType,
        `>`,
    ]
]

comptime field_decorator_of[T: AnyType, idx: Int, D: AnyType] = ParameterList[
    type=D,
    __mlir_attr[
        `#kgen.struct_field_decorator_of<`,
        T,
        `, `,
        idx.__mlir_index__(),
        `, `,
        D,
        `> : !kgen.param_list<`,
        +D,
        `>`,
    ],
]

comptime field_decorator_of_by_name[
    T: AnyType, name: StringLiteral, D: AnyType
] = ParameterList[
    type=D,
    __mlir_attr[
        `#kgen.struct_field_decorator_of<`,
        T,
        `, #kgen.struct_field_index_by_name<`,
        T,
        `, `,
        name.value,
        `> : index, `,
        D,
        `> : !kgen.param_list<`,
        +D,
        `>`,
    ],
]

# ===----------------------------------------------------------------------=== #
# Decorators and decorated types
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct tag(Decorator):
    pass


# A second fieldless decorator: its value lowers to a constant that is
# byte-identical to `tag()`'s, so it is the regression case for identity.
@fieldwise_init
struct latetag(Decorator):
    pass


struct serde(Decorator):
    var rename: StaticString
    var skip: Bool

    def __init__(out self, rename: StaticString = "", skip: Bool = False):
        self.rename = rename
        self.skip = skip


@tag
struct Thing(Movable where False):
    @serde(rename="NAME")
    var name: Int
    var plain: Int


struct Parametric[N: StringLiteral](Movable where False):
    @serde(rename=Self.N)
    var x: Int


@serde(rename="NAME")
@tag
struct Stacked(Movable where False):
    var x: Int


struct Plain(Movable where False):
    var x: Int


# Carries the *other* fieldless decorator, for the enumeration cross-check
# below: `@tag` and `@latetag` values are byte-identical once lowered, so the
# only thing that can tell `LateThing`'s decorator from `Thing`'s is the
# recorded identity.
@latetag
struct LateThing(Movable where False):
    var x: Int


# A decorator struct that declares one parameter. It is never spelled at the
# use site: the compiler binds it to the type of the decorated declaration,
# so `@tagged` on an `Int` field is `tagged[Int]` no matter which constructor
# arguments were written. Identity stays symbol-based (`tagged`), and the
# parameterization is checked against the site -- see `packDecorators`.
struct tagged[FieldT: AnyType](Decorator):
    var note: StaticString

    def __init__(out self, note: StaticString = ""):
        self.note = note


@tagged("outer")
struct BoundThing(Movable where False):
    @tagged("a")
    var x: Int

    @tagged("s")
    var s: StaticString


# The only shape where the stored parameterization and the queried one are
# spelled in different worlds: the field's type is the enclosing struct's own
# parameter, so the decorator is stored as `tagged[T]` and the folder has to
# rebind it through the struct's parameter values before it can be compared
# with a `tagged[Int]` query. For `BoundThing` above -- and every other
# fixture -- that rebinding step is an identity function.
#
# The struct-level decorator here is the self-referential half:
# `tagged[DependentThing[T]]`, rebuilt while evaluating an attribute *on*
# `DependentThing`.
@tagged("outer")
struct DependentThing[T: Movable & Deinitable](Movable where False):
    @tagged("dep")
    var x: Self.T


# CHECK-LABEL: lit.fn @"main()"
def main():
    # `decorator_types` yields the type of each stored decorator value, in
    # application order (nearest to the declaration first).
    # CHECK: lit.alias.decl *"structDecTypes{{.*}}param_list<!AnyType> [!tag, !serde]>
    comptime structDecTypes = decorator_types[Stacked]()
    # CHECK: lit.alias.decl *"structDecType0`{{[0-9]*}}": !AnyType = <!tag>
    comptime structDecType0 = structDecTypes[0]
    # CHECK: lit.alias.decl *"structDecType1`{{[0-9]*}}": !AnyType = <!serde>
    comptime structDecType1 = structDecTypes[1]

    # Undecorated struct: empty list.
    # CHECK: lit.alias.decl *"plainDecTypes{{.*}}param_list<!AnyType> []>
    comptime plainDecTypes = decorator_types[Plain]()

    # `decorator_of` selects by decorator type, and folds to a param_list of
    # that type holding the single matching value.
    # CHECK: lit.alias.decl *"taggedOf`{{[0-9]*}}": !lit.struct<#ParameterList <:!AnyType !tag, :param_list<!tag> [{{.*}}@tag::@"__init__()"{{.*}}]>>
    comptime taggedOf = decorator_of[Thing, tag]()
    # CHECK: lit.alias.decl *"taggedOfSize{{.*}}scalar<index> 1
    comptime taggedOfSize = taggedOf.size

    # A different fieldless decorator does not match, even though `latetag()`
    # and `tag()` fold to byte-identical values once lowered: the match is on
    # the decorator's base struct, not on the value.
    # CHECK: lit.alias.decl *"lateOfSize{{.*}}param_list<!latetag> []>{{.*}}scalar<index> 0
    comptime lateOfSize = decorator_of[Thing, latetag]().size

    # A decorator carrying a payload folds to the value that was written.
    # CHECK: lit.alias.decl *"serdeOf0`{{[0-9]*}}": !serde = <{{.*}}"NAME"{{.*}}>
    comptime serdeOf0 = decorator_of[Stacked, serde]()[0]

    # Field decorators, by index and by name.
    # CHECK: lit.alias.decl *"fieldDecTypes0{{.*}}param_list<!AnyType> [!serde]>
    comptime fieldDecTypes0 = field_decorator_types[Thing, 0]()
    # CHECK: lit.alias.decl *"fieldDecType0`{{[0-9]*}}": !AnyType = <!serde>
    comptime fieldDecType0 = fieldDecTypes0[0]
    # CHECK: lit.alias.decl *"plainFieldDecTypes{{.*}}param_list<!AnyType> []>
    comptime plainFieldDecTypes = field_decorator_types[Thing, 1]()
    # CHECK: lit.alias.decl *"fieldOf0`{{[0-9]*}}": !serde = <{{.*}}"NAME"{{.*}}>
    comptime fieldOf0 = field_decorator_of[Thing, 0, serde]()[0]
    # By name, via a nested struct_field_index_by_name operand.
    # CHECK: lit.alias.decl *"byNameOf0`{{[0-9]*}}": !serde = <{{.*}}"NAME"{{.*}}>
    comptime byNameOf0 = field_decorator_of_by_name[Thing, "name", serde]()[0]
    # Undecorated field: empty list.
    # CHECK: lit.alias.decl *"plainFieldOfSize{{.*}}param_list<!serde> []>{{.*}}scalar<index> 0
    comptime plainFieldOfSize = field_decorator_of[Thing, 1, serde]().size

    # Parameter references in a decorator payload are rebound per
    # instantiation.
    # CHECK: lit.alias.decl *"paramOf0`{{[0-9]*}}": !serde = <{{.*}}"bound"{{.*}}>
    comptime paramOf0 = field_decorator_of[Parametric["bound"], 0, serde]()[0]

    # A parameterized decorator is queried by naming the parameterization the
    # site derives. Matching one folds to the stored value...
    # CHECK: lit.alias.decl *"boundOf0`{{[0-9]*}}": !lit.struct<#tagged <:!AnyType !Int>> = <{{.*}}"a"{{.*}}>
    comptime boundOf0 = field_decorator_of[BoundThing, 0, tagged[Int]]()[0]

    # ...and naming the *wrong* parameterization matches nothing, even though
    # the base struct symbol -- all identity records -- is the same `tagged`.
    # CHECK: lit.alias.decl *"boundWrongSize{{.*}}param_list<!lit.struct<#tagged <:!AnyType @{{.*}}StringSpan{{.*}}>>> []>{{.*}}scalar<index> 0
    comptime boundWrongSize = field_decorator_of[
        BoundThing, 0, tagged[StaticString]
    ]().size

    # The other field's own binding does match its own spelling.
    # CHECK: lit.alias.decl *"boundStrSize{{.*}}scalar<index> 1
    comptime boundStrSize = field_decorator_of[
        BoundThing, 1, tagged[StaticString]
    ]().size

    # Struct-level: the binding is the struct's own type.
    # CHECK: lit.alias.decl *"structBoundSize{{.*}}scalar<index> 1
    comptime structBoundSize = decorator_of[
        BoundThing, tagged[BoundThing]
    ]().size
    # CHECK: lit.alias.decl *"structBoundWrongSize{{.*}}scalar<index> 0
    comptime structBoundWrongSize = decorator_of[BoundThing, tagged[Int]]().size

    # `decorator_types` reports the *bound* type, so its output is a legal
    # `decorator_of` query -- the enumerate-then-fetch round trip.
    # CHECK: lit.alias.decl *"boundDecTypes{{.*}}param_list<!AnyType> [@{{.*}}::@tagged<:!AnyType !Int>]>
    comptime boundDecTypes = field_decorator_types[BoundThing, 0]()

    # Rebinding path, at parse time: the stored `tagged[T]` becomes
    # `tagged[Int]` only because the site type is rebound through
    # `DependentThing`'s parameter values.
    # CHECK: lit.alias.decl *"depOfSize{{.*}}scalar<index> 1
    comptime depOfSize = field_decorator_of[
        DependentThing[Int], 0, tagged[Int]
    ]().size
    # CHECK: lit.alias.decl *"depWrongSize{{.*}}scalar<index> 0
    comptime depWrongSize = field_decorator_of[
        DependentThing[Int], 0, tagged[StaticString]
    ]().size
    # A different instantiation of the same declaration, a different type.
    # CHECK: lit.alias.decl *"depStrSize{{.*}}scalar<index> 1
    comptime depStrSize = field_decorator_of[
        DependentThing[StaticString], 0, tagged[StaticString]
    ]().size
    # CHECK: lit.alias.decl *"depDecTypes{{.*}}param_list<!AnyType> [@{{.*}}::@tagged<:!AnyType !Int>]>
    comptime depDecTypes = field_decorator_types[DependentThing[Int], 0]()

    # Struct level on the same declaration: the binding is the parameterized
    # self type, so it is rebound per instantiation as well.
    # CHECK: lit.alias.decl *"depStructSize{{.*}}scalar<index> 1
    comptime depStructSize = decorator_of[
        DependentThing[Int], tagged[DependentThing[Int]]
    ]().size
    # CHECK: lit.alias.decl *"depStructWrongSize{{.*}}scalar<index> 0
    comptime depStructWrongSize = decorator_of[
        DependentThing[Int], tagged[DependentThing[StaticString]]
    ]().size


# ===----------------------------------------------------------------------=== #
# Elaboration path
# ===----------------------------------------------------------------------=== #

# Everything above folds while parsing, against `lit.struct.decl`. Inside a
# generic function nothing can: `T` is symbolic, so the attributes survive into
# the IR and are folded by the elaborator instead, against the lowered
# `kgen.struct.generator`. That is a different resolution path -- the type
# values are `#kgen.instref`s rather than LIT struct types -- and a different
# identity source, because lowering anonymizes decorator value types and only
# the parallel `decoratorTypeNames` array still says which decorator struct
# produced each value.
#
# Each helper below is instantiated exactly once, from the correspondingly
# named `@export` probe, so its concretized body is uniquely identified by the
# helper's own name.


# `@tag` is present, so the query folds to a one-element list.
# ELAB-LABEL: kgen.func @"{{.*}}tagged_tag_count
# ELAB: kgen.param.constant: scalar<index> = <1>
def tagged_tag_count[T: AnyType]() -> Int:
    return decorator_of[T, tag]().size


# An undecorated struct folds to the empty list.
# ELAB-LABEL: kgen.func @"{{.*}}plain_tag_count
# ELAB: kgen.param.constant: scalar<index> = <0>
def plain_tag_count[T: AnyType]() -> Int:
    return decorator_of[T, tag]().size


# The identity regression case, post-lowering: `Thing` carries `@tag`, whose
# value has by now been anonymized to a constant byte-identical to `latetag()`'s.
# Only `decoratorTypeNames` can tell them apart, so this must be empty.
# ELAB-LABEL: kgen.func @"{{.*}}tagged_latetag_count
# ELAB: kgen.param.constant: scalar<index> = <0>
def tagged_latetag_count[T: AnyType]() -> Int:
    return decorator_of[T, latetag]().size


# Field variant, decorated field.
# ELAB-LABEL: kgen.func @"{{.*}}decorated_field_serde_count
# ELAB: kgen.param.constant: scalar<index> = <1>
def decorated_field_serde_count[T: AnyType]() -> Int:
    return field_decorator_of[T, 0, serde]().size


# Field variant, undecorated field.
# ELAB-LABEL: kgen.func @"{{.*}}plain_field_serde_count
# ELAB: kgen.param.constant: scalar<index> = <0>
def plain_field_serde_count[T: AnyType]() -> Int:
    return field_decorator_of[T, 1, serde]().size


# `decorator_types` must report the decorator's *named* type. Post-lowering the
# stored value's own type has been anonymized to `!kgen.struct<() memoryOnly>`,
# which is byte-identical for any two fieldless decorators, so reporting it
# would make this comparison false -- and `tagged_first_is_latetag` below true.
# ELAB-LABEL: kgen.func @"{{.*}}tagged_first_is_tag
# ELAB: kgen.param.constant: scalar<index> = <1>
def tagged_first_is_tag[T: AnyType]() -> Int:
    comptime tys = decorator_types[T]()
    comptime if tys[0] == tag:
        return 1
    else:
        return 0


# The same enumeration must not answer to a different decorator struct.
# ELAB-LABEL: kgen.func @"{{.*}}tagged_first_is_latetag
# ELAB: kgen.param.constant: scalar<index> = <0>
def tagged_first_is_latetag[T: AnyType]() -> Int:
    comptime tys = decorator_types[T]()
    comptime if tys[0] == latetag:
        return 1
    else:
        return 0


# Two structs carrying *different* fieldless decorators must not enumerate
# equal types -- the sharpest form of the same check, since it never mentions
# a decorator by name and so cannot pass by both sides being anonymous.
# ELAB-LABEL: kgen.func @"{{.*}}first_types_match
# ELAB: kgen.param.constant: scalar<index> = <0>
def first_types_match[T: AnyType, U: AnyType]() -> Int:
    comptime a = decorator_types[T]()
    comptime b = decorator_types[U]()
    comptime if a[0] == b[0]:
        return 1
    else:
        return 0


# The field variant reports named types too.
# ELAB-LABEL: kgen.func @"{{.*}}field_first_is_serde
# ELAB: kgen.param.constant: scalar<index> = <1>
def field_first_is_serde[T: AnyType]() -> Int:
    comptime tys = field_decorator_types[T, 0]()
    comptime if tys[0] == serde:
        return 1
    else:
        return 0


# The queried decorator type may itself be generic. While parsing it stays
# symbolic, which must leave the attribute unfolded and *silent* rather than
# diagnosed, and resolve once the caller binds it.
# ELAB-LABEL: kgen.func @"{{.*}}generic_query_count
# ELAB: kgen.param.constant: scalar<index> = <1>
#
# Nothing may survive unfolded: before the query operand could be decoded on
# this path, `#kgen.struct_decorator_of` was reproduced verbatim into the
# elaborated IR, with no diagnostic at all.
# ELAB-NOT: struct_decorator_of
# ELAB-NOT: struct_decorator_types
def generic_query_count[D: AnyType]() -> Int:
    return decorator_of[Thing, D]().size


# Post-lowering, the stored value's type is anonymized and only the base
# struct symbol survives in `decoratorTypeNames` -- which is the same symbol
# for `tagged[Int]` and `tagged[StaticString]`. The parameterization check
# is what keeps these two apart, comparing the query against the *site's*
# type rather than against anything recorded per value.
# ELAB-LABEL: kgen.func @"{{.*}}bound_field_int_count
# ELAB: kgen.param.constant: scalar<index> = <1>
def bound_field_int_count[T: AnyType]() -> Int:
    return field_decorator_of[T, 0, tagged[Int]]().size


# ELAB-LABEL: kgen.func @"{{.*}}bound_field_wrong_count
# ELAB: kgen.param.constant: scalar<index> = <0>
def bound_field_wrong_count[T: AnyType]() -> Int:
    return field_decorator_of[T, 0, tagged[StaticString]]().size


# ELAB-LABEL: kgen.func @"{{.*}}bound_struct_count
# ELAB: kgen.param.constant: scalar<index> = <1>
def bound_struct_count[T: AnyType]() -> Int:
    return decorator_of[T, tagged[BoundThing]]().size


# ELAB-LABEL: kgen.func @"{{.*}}bound_struct_wrong_count
# ELAB: kgen.param.constant: scalar<index> = <0>
def bound_struct_wrong_count[T: AnyType]() -> Int:
    return decorator_of[T, tagged[Int]]().size


# `decorator_types` rebuilds the reported type from the recorded symbol, so it
# has to rebind the site-derived parameter too: an arity-mismatched `genref`
# is not merely a wrong answer here, it asserts in name mangling.
# ELAB-LABEL: kgen.func @"{{.*}}bound_field_first_is_tagged_int
# ELAB: kgen.param.constant: scalar<index> = <1>
def bound_field_first_is_tagged_int[T: AnyType]() -> Int:
    comptime tys = field_decorator_types[T, 0]()
    comptime if tys[0] == tagged[Int]:
        return 1
    else:
        return 0


# ELAB-LABEL: kgen.func @"{{.*}}bound_field_first_is_tagged_str
# ELAB: kgen.param.constant: scalar<index> = <0>
def bound_field_first_is_tagged_str[T: AnyType]() -> Int:
    comptime tys = field_decorator_types[T, 0]()
    comptime if tys[0] == tagged[StaticString]:
        return 1
    else:
        return 0


# The rebinding path again, post-lowering: `T` is symbolic here, so the
# attributes survive into the IR and the elaborator does the rebinding.
# ELAB-LABEL: kgen.func @"{{.*}}dep_field_count
# ELAB: kgen.param.constant: scalar<index> = <1>
def dep_field_count[T: AnyType, Q: AnyType]() -> Int:
    return field_decorator_of[T, 0, tagged[Q]]().size


# ELAB-LABEL: kgen.func @"{{.*}}dep_field_wrong_count
# ELAB: kgen.param.constant: scalar<index> = <0>
def dep_field_wrong_count[T: AnyType, Q: AnyType]() -> Int:
    return field_decorator_of[T, 0, tagged[Q]]().size


# `decorator_types` on the rebinding path: the reported type is rebuilt from
# the recorded symbol plus the *rebound* site type.
# ELAB-LABEL: kgen.func @"{{.*}}dep_field_first_is_tagged_int
# ELAB: kgen.param.constant: scalar<index> = <1>
def dep_field_first_is_tagged_int[T: AnyType]() -> Int:
    comptime tys = field_decorator_types[T, 0]()
    comptime if tys[0] == tagged[Int]:
        return 1
    else:
        return 0


# Struct level, self-referential: rebuilding `tagged[DependentThing[Int]]`
# while evaluating an attribute on `DependentThing[Int]`.
# ELAB-LABEL: kgen.func @"{{.*}}dep_struct_first_is_self
# ELAB: kgen.param.constant: scalar<index> = <1>
def dep_struct_first_is_self[T: AnyType]() -> Int:
    comptime tys = decorator_types[T]()
    comptime if tys[0] == tagged[DependentThing[Int]]:
        return 1
    else:
        return 0


@export
def probe_tagged_tag() abi("C") -> Int:
    return tagged_tag_count[Thing]()


@export
def probe_plain_tag() abi("C") -> Int:
    return plain_tag_count[Plain]()


@export
def probe_tagged_latetag() abi("C") -> Int:
    return tagged_latetag_count[Thing]()


@export
def probe_decorated_field_serde() abi("C") -> Int:
    return decorated_field_serde_count[Thing]()


@export
def probe_plain_field_serde() abi("C") -> Int:
    return plain_field_serde_count[Thing]()


@export
def probe_tagged_first_is_tag() abi("C") -> Int:
    return tagged_first_is_tag[Thing]()


@export
def probe_tagged_first_is_latetag() abi("C") -> Int:
    return tagged_first_is_latetag[Thing]()


@export
def probe_first_types_match() abi("C") -> Int:
    return first_types_match[Thing, LateThing]()


@export
def probe_field_first_is_serde() abi("C") -> Int:
    return field_first_is_serde[Thing]()


@export
def probe_generic_query() abi("C") -> Int:
    return generic_query_count[tag]()


@export
def probe_bound_field_int() abi("C") -> Int:
    return bound_field_int_count[BoundThing]()


@export
def probe_bound_field_wrong() abi("C") -> Int:
    return bound_field_wrong_count[BoundThing]()


@export
def probe_bound_struct() abi("C") -> Int:
    return bound_struct_count[BoundThing]()


@export
def probe_bound_struct_wrong() abi("C") -> Int:
    return bound_struct_wrong_count[BoundThing]()


@export
def probe_bound_field_first_is_tagged_int() abi("C") -> Int:
    return bound_field_first_is_tagged_int[BoundThing]()


@export
def probe_bound_field_first_is_tagged_str() abi("C") -> Int:
    return bound_field_first_is_tagged_str[BoundThing]()


@export
def probe_dep_field() abi("C") -> Int:
    return dep_field_count[DependentThing[Int], Int]()


@export
def probe_dep_field_wrong() abi("C") -> Int:
    return dep_field_wrong_count[DependentThing[Int], StaticString]()


@export
def probe_dep_field_first_is_tagged_int() abi("C") -> Int:
    return dep_field_first_is_tagged_int[DependentThing[Int]]()


@export
def probe_dep_struct_first_is_self() abi("C") -> Int:
    return dep_struct_first_is_self[DependentThing[Int]]()
