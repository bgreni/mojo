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

# The *negative* half of the site-derived parameter bound (Gap 2), against the
# real stdlib. The positive half -- a decorator whose parameter is bound
# `AnyType` attaching to a `List` field -- is
# `test_collection_field_decorator` in
# `mojo/stdlib/test/reflection/test_decorators.mojo`.
#
# The point being pinned: a decorator's parameter bound is checked at every
# declaration it is attached to, *whatever arguments were passed*, because the
# binding is site-derived rather than inferred. So an over-strong bound is not
# a constraint on the payload, it is a constraint on which fields the
# decorator may decorate at all. `List` and `Dict` conform to `Copyable` but
# not `ImplicitlyCopyable`, which makes `ImplicitlyCopyable` the specific
# bound that silently excludes every collection field. This is easy to write
# by accident (it is what a naive `self.default = default` forces) and nothing
# else in the suite would catch it.
#
# Both cases are declaration-level (a struct or a field), which `comptime if`
# cannot gate, so this whole file is a single expected-to-fail compile and one
# FileCheck run reads both diagnostics.

# RUN: not kgen %s -elaborate 2>&1 | FileCheck %s


# An over-strong bound. Everything about the payload here would be happy with
# `AnyType`; only the bound rejects the field.
struct too_strong[FieldT: ImplicitlyCopyable & Deinitable](Decorator):
    var rename: Optional[StaticString]

    def __init__(out self, rename: Optional[StaticString] = None):
        self.rename = rename


struct HasListField:
    # Note the argument passed is only `rename` -- the decorator never touches
    # a value of `FieldT`. The bound is still checked, because the binding
    # comes from the field, not from the arguments.
    # CHECK: 'too_strong' parameter 'FieldT' has 'ImplicitlyCopyable & Deinitable' type, but value has type 'AnyStruct[List[Int]]'
    @too_strong(rename=StaticString("xs"))
    var xs: List[Int]


# The other end of the range: storing a *value* of `FieldT` cannot go all the
# way down to `AnyType`, even taking the argument `var` and transferring it.
# `Movable & Deinitable` is the floor, and it is enough -- `List` satisfies it,
# which is what `test_collection_field_decorator` relies on.
# CHECK: field 'default' has non-'Deinitable' type 'Optional[FieldT]'
struct too_weak[FieldT: AnyType](Decorator):
    var default: Optional[Self.FieldT]

    def __init__(out self, var default: Optional[Self.FieldT] = None):
        self.default = default^


def main():
    pass
