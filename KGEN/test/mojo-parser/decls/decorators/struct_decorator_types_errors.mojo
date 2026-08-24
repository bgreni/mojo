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
# RUN: %parse-mojo-isolated -verify-diagnostics %s


# The `BadKeyword` case below attaches these notes: `serde` has two
# constructor candidates (the one below and the auto-generated move
# constructor), so a resolution failure reports one note per candidate.
# expected-note @+2 {{candidate not viable: missing required argument: 'move'}}
# expected-note @+1 {{def __init__(out self, *, deinit move: Self)}}
struct serde(Decorator):
    var rename: StaticString
    var skip: Bool

    # expected-note @+1 {{candidate not viable: unexpected keyword argument 'renmae'}}
    def __init__(out self, rename: StaticString = "", skip: Bool = False):
        self.rename = rename
        self.skip = skip


# `NotADecorator` needs a working bare constructor so that the errors below
# exercise the `Decorator`-conformance check rather than tripping over a
# missing constructor argument first: a decorator value is constructed before
# it is checked for conformance.
@fieldwise_init
struct NotADecorator:
    pass


# A struct that does not conform to `Decorator` cannot be used as one.
# expected-error @+1 {{'NotADecorator' is not a decorator; decorator structs must conform to 'Decorator'}}
@NotADecorator
struct BadStructDecorator(Movable where False):
    pass


# Unknown names keep today's diagnostic.
# expected-error @+1 {{use of unknown declaration 'serdee'}}
@serdee(rename="x")
struct Typo(Movable where False):
    pass


# Arguments go through ordinary constructor overload resolution: `serde` now
# has two candidates (the user-written `__init__` and the auto-generated
# move constructor), so a bad keyword surfaces as a resolution failure with
# one note per rejected candidate.
# expected-error @+1 {{no matching function in initialization}}
@serde(renmae="x")
struct BadKeyword(Movable where False):
    pass


# Decorator structs are not accepted on functions: the existing
# compiler-decorator path rejects the value as before.
# expected-error @+1 {{unsupported compiler decorator}}
@serde
def decorated_function():
    pass


# The call form on a function is likewise not a decorator value; it keeps the
# pre-existing diagnostic from resolving `serde(...)` as a constructor call
# whose result is then rejected by the compiler-decorator allowlist.
# expected-error @+1 {{unsupported compiler decorator}}
@serde(skip=True)
def called_decorated_function():
    pass


# The bare form constructs with no arguments, so a decorator with a required
# constructor argument cannot be spelled bare: neither the generated
# constructor nor the auto-generated move constructor accepts zero arguments,
# so resolution fails with one note per candidate.
# expected-note @+5 {{candidate not viable: missing required argument: 'move'}}
# expected-note @+4 {{def __init__(out self, *, deinit move: Self)}}
# expected-note @+3 {{candidate not viable: missing required argument: 'required'}}
# expected-note @+2 {{def __init__(out self, required: StaticString)}}
@fieldwise_init
struct needs_arg(Decorator):
    var required: StaticString


# expected-error @+1 {{no matching function in initialization}}
@needs_arg
struct BareNeedsArg(Movable where False):
    pass


# NOTE: the payload-legality case (a decorator field that isn't a legal
# comptime parameter value, e.g. a heap-allocating `String`) and the
# "cannot use a dynamic value in a parameter list" case (binding a
# decorator's parameter to a runtime value) are both pre-existing binder
# diagnostics that cannot be exercised in this isolated-parser mode: the
# `String`/`List` stubs available here are no-ops that never actually
# allocate, so constructing a decorator from one folds trivially instead of
# tripping the real (heap-allocating) implementation's binder failure, and
# exercising the parameter-list case needs a decorator struct in scope of a
# function with a runtime argument, which this mode's isolation (no nested
# structs, no module-level globals) cannot set up.


# A subscript whose base is a *function* is not a decorator value and keeps the
# existing parameter-binding diagnostic.
# expected-note @+1 {{function declared here}}
def register(a: StringLiteral):
    return


# expected-error @+1 {{unexpected parameter}}
@register["x"]
struct FnSubscript(Movable where False):
    pass


# An overloaded function decorator is not a decorator value either: it is
# overload-resolved and rejected by the compiler-decorator allowlist as before.
def twice(a: Int):
    return


def twice(a: StringLiteral):
    return


# expected-error @+1 {{unsupported compiler decorator}}
@twice(1)
struct OverloadedFnDecorator(Movable where False):
    pass


struct FieldErrors(Movable where False):
    # expected-error @+1 {{'NotADecorator' is not a decorator; decorator structs must conform to 'Decorator'}}
    @NotADecorator
    var a: Int

    # Function decorators are still not supported on fields.
    # expected-error @+1 {{decorators not supported on this statement}}
    @register("x")
    var c: Int

    # Unknown names on fields keep today's diagnostic.
    # expected-error @+1 {{decorators not supported on this statement}}
    @stable
    var d: Int


# ===----------------------------------------------------------------------=== #
# Decorator structs may declare at most one parameter
# ===----------------------------------------------------------------------=== #

# A decorator is identified by its struct, and that identity has to survive
# lowering, which anonymizes the value's type and leaves only a symbol naming
# the struct. A symbol cannot distinguish one parameter binding from another.
#
# One parameter is nevertheless legal, because the *compiler* picks its
# binding: always the type of the decorated declaration (see
# `struct_decorator_types.mojo`). Symbol plus site therefore still
# reconstructs the parameterization exactly. A second parameter has no
# site-derived value to bind, so it is rejected.


@fieldwise_init
struct two_params[A: AnyType = NoneType, B: Int = 0](Decorator):
    pass


# expected-error @+1 {{decorator struct 'two_params' may declare at most one parameter; the one parameter is bound to the decorated declaration's type, so carry any other payload in fields instead (a type payload can be passed as a function value)}}
@two_params
struct UsesTwoParams(Movable where False):
    pass


# The bracket form is rejected for the same reason.
# expected-error @+1 {{decorator struct 'two_params' may declare at most one parameter}}
@two_params[A=String, B=1]
struct UsesTwoParamsBracket(Movable where False):
    pass


struct TwoParamsField(Movable where False):
    # expected-error @+1 {{decorator struct 'two_params' may declare at most one parameter}}
    @two_params
    var x: Int


# *Undefaulted* parameters reach the same diagnostic. They have to be caught
# before the decorator value is emitted: emission would fail parameter
# inference first and report `'two_undefaulted[_]' is not concrete` instead.
# (One undefaulted parameter is fine -- the site-derived binding supplies it.)
@fieldwise_init
struct two_undefaulted[A: AnyType, B: AnyType](Decorator):
    pass


# expected-error @+1 {{decorator struct 'two_undefaulted' may declare at most one parameter; the one parameter is bound to the decorated declaration's type, so carry any other payload in fields instead (a type payload can be passed as a function value)}}
@two_undefaulted
struct UsesTwoUndefaulted(Movable where False):
    pass


struct TwoUndefaultedField(Movable where False):
    # expected-error @+1 {{decorator struct 'two_undefaulted' may declare at most one parameter}}
    @two_undefaulted
    var x: Int


# A *conditional* `Decorator` conformance is unprovable from the declaration
# alone -- there are no parameter bindings to evaluate `N > 0` against -- so
# the pre-emission check cannot fire on it. The post-emission check, which has
# the bindings the emitted value supplies, catches it instead. Without that
# second check this struct is silently accepted as a two-parameter decorator.
# Both parameters are defaulted so that emission itself succeeds and the
# post-emission check is the thing that fires.
@fieldwise_init
struct conditional[A: AnyType = NoneType, N: Int = 1](Decorator where N > 0):
    pass


# expected-error @+1 {{decorator struct 'conditional' may declare at most one parameter; the one parameter is bound to the decorated declaration's type, so carry any other payload in fields instead (a type payload can be passed as a function value)}}
@conditional
struct UsesConditional(Movable where False):
    pass


struct ConditionalField(Movable where False):
    # The field-level path shares the exact same post-emission check as the
    # struct-level path above (`tryEmitDecoratorValue` is not duplicated
    # per call site) -- this case pins that it actually fires there too.
    # expected-error @+1 {{decorator struct 'conditional' may declare at most one parameter}}
    @conditional
    var x: Int


# A parameterized struct that is *not* a decorator keeps its own diagnostics
# rather than being told it is a malformed decorator. With one parameter the
# site-derived binding happens first, so what is reported is still the
# conformance failure, not a binding failure.
@fieldwise_init
struct ParameterizedNonDecorator[Coerce: AnyType = NoneType]:
    pass


# expected-error @+1 {{'ParameterizedNonDecorator' is not a decorator; decorator structs must conform to 'Decorator'}}
@ParameterizedNonDecorator
struct UsesParameterizedNonDecorator(Movable where False):
    pass


# ===----------------------------------------------------------------------=== #
# The one parameter may not be spelled at the use site
# ===----------------------------------------------------------------------=== #

# Binding is site-derived precisely so that every occurrence of a decorator on
# one declaration has the same type. Letting the use site override the binding
# would reintroduce the instability: `@one_param[Int]` and `@one_param` on the
# same field would be different types, and a reader querying one spelling
# would silently miss the other.


struct one_param[FieldT: AnyType](Decorator):
    var note: StaticString

    def __init__(out self, note: StaticString = ""):
        self.note = note


# expected-error @+1 {{decorator struct 'one_param' does not accept an explicit parameter list; its parameter is bound to the decorated declaration's type}}
@one_param[Int]
struct ExplicitBinding(Movable where False):
    pass


struct ExplicitBindingField(Movable where False):
    # expected-error @+1 {{decorator struct 'one_param' does not accept an explicit parameter list}}
    @one_param[Int]("x")
    var x: Int


# The rejection happens *after* emission settles conformance, which is what
# keeps it from stealing the diagnostic a parameterized struct that is not a
# decorator is owed. Pin both sides of that: a non-decorator with brackets
# still reports "is not a decorator"...
# expected-error @+1 {{'ParameterizedNonDecorator' is not a decorator; decorator structs must conform to 'Decorator'}}
@ParameterizedNonDecorator[Coerce=String]
struct UsesParameterizedNonDecoratorBracket(Movable where False):
    pass


# ...and a *conditional* `Decorator` conformance, which cannot be proven from
# the declaration alone, is still caught. Deferring to after emission is what
# makes one guard enough here, unlike the parameter-count rule which needs
# two: the bracketed form binds its own parameters, so it always emits.
@fieldwise_init
struct cond_one[N: Int = 1](Decorator where N > 0):
    pass


# expected-error @+1 {{decorator struct 'cond_one' does not accept an explicit parameter list; its parameter is bound to the decorated declaration's type}}
@cond_one[N=2]
struct CondBracket(Movable where False):
    pass


# The no-duplicates rule is unchanged by parameterization: `dedupeDecorators`
# keys on the base struct alone, and that is still exact, because both
# occurrences are on the *same* declaration and therefore share the same
# site-derived binding -- same base struct still implies same type here.
struct DuplicateParameterized(Movable where False):
    # expected-note @+2 {{previous decorator here}}
    # expected-error @+2 {{duplicate decorator 'one_param'}}
    @one_param("a")
    @one_param("b")
    var x: Int


# expected-note @+2 {{previous decorator here}}
# expected-error @+2 {{duplicate decorator 'one_param'}}
@one_param("a")
@one_param("b")
struct DuplicateParameterizedStruct(Movable where False):
    pass


# ===----------------------------------------------------------------------=== #
# The one parameter must be able to take a type
# ===----------------------------------------------------------------------=== #

# The arity rule is "at most one", not "at most one type parameter": the
# binding path is taken on parameter *count* alone. A parameter that cannot
# hold a type is therefore rejected by ordinary parameter binding rather than
# by a decorator-specific check. The diagnostic is still specific enough to
# act on -- it names the decorator, the parameter, its declared type, and the
# site type that could not be bound to it -- so this is pinned as the
# behavior rather than given a check of its own.


# The rejections below point their note back here, at the declaration.
# expected-note @+2 {{'valueparam' declared here}}
@fieldwise_init
struct valueparam[N: Int = 0](Decorator):
    pass


# expected-error @+1 {{'valueparam' parameter 'N' has 'Int' type, but value has type 'AnyStruct[UsesValueParam]'}}
@valueparam
struct UsesValueParam(Movable where False):
    pass


struct ValueParamField(Movable where False):
    # The site type here is the *field's*, so the message differs from the
    # struct-level one above -- which is itself evidence that the binding is
    # site-derived on both paths.
    # expected-error @+1 {{'valueparam' parameter 'N' has 'Int' type, but value has type 'AnyStruct[Int]'}}
    @valueparam
    var x: Int


# A variadic pack does not slip through the arity check: it carries more than
# one parameter declaration, so it lands on the "at most one" rejection rather
# than on the binding path. Checked rather than assumed.
@fieldwise_init
struct packparam[*Ts: AnyType](Decorator):
    pass


# expected-error @+1 {{decorator struct 'packparam' may declare at most one parameter}}
@packparam
struct UsesPack(Movable where False):
    pass


# ===----------------------------------------------------------------------=== #
# Duplicate decorators
# ===----------------------------------------------------------------------=== #

# At most one decorator per base struct is allowed on a declaration. The error
# lands on the second occurrence going down the page, with the note pointing
# back up at the first.
# expected-note @+2 {{previous decorator here}}
# expected-error @+2 {{duplicate decorator 'serde'}}
@serde(rename="a")
@serde(rename="b")
struct DuplicateSerde(Movable where False):
    pass


struct DuplicateField(Movable where False):
    # expected-note @+2 {{previous decorator here}}
    # expected-error @+2 {{duplicate decorator 'serde'}}
    @serde(rename="a")
    @serde(rename="b")
    var x: Int


# ===----------------------------------------------------------------------=== #
# RepeatableDecorator does not exempt other decorators from the duplicate check
# ===----------------------------------------------------------------------=== #

# The no-duplicates rule stays the default: a decorator opts out only by
# conforming to `RepeatableDecorator`, and the exemption is keyed on that
# specific base struct, not on "some decorator on this declaration happens to
# be repeatable." This must not regress -- it is the guarantee every existing
# `decorator_of[D]()` caller relies on.


struct label(RepeatableDecorator):
    var wire: StaticString

    def __init__(out self, wire: StaticString = ""):
        self.wire = wire


# `label` repeats freely (no `expected-error` here means the parser must not
# diagnose it), but the *non*-repeatable `serde` stacked alongside it still
# triggers the duplicate check exactly as `DuplicateSerde` above does.
# expected-note @+2 {{previous decorator here}}
# expected-error @+2 {{duplicate decorator 'serde'}}
@serde(rename="x")
@serde(rename="y")
@label("a")
@label("b")
struct MixedRepeatableAndDuplicateNonRepeatable(Movable where False):
    pass


struct MixedRepeatableAndDuplicateNonRepeatableField(Movable where False):
    # expected-note @+2 {{previous decorator here}}
    # expected-error @+2 {{duplicate decorator 'serde'}}
    @serde(rename="x")
    @serde(rename="y")
    @label("a")
    @label("b")
    var x: Int
