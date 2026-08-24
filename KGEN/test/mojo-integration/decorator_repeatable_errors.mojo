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

# Compile-time errors for the RepeatableDecorator opt-in (Gap 4), against the
# real stdlib reflection API. Each case is gated by `get_defined_bool` the way
# `struct_field_reflection_errors.mojo` gates its cases: a hard compile error
# must not poison every other case in the file, so each lives behind its own
# flag and its own RUN line.

# RUN: not kgen %s -elaborate -D TEST_DECORATOR_OF_REPEATABLE_STRUCT=1 2>&1 | FileCheck %s --check-prefix=CHECK-STRUCT
# RUN: not kgen %s -elaborate -D TEST_DECORATOR_OF_REPEATABLE_FIELD=1 2>&1 | FileCheck %s --check-prefix=CHECK-FIELD
# RUN: not kgen %s -elaborate -D TEST_FOLDER_HARDENING_STRUCT=1 2>&1 | FileCheck %s --check-prefix=CHECK-FOLDER-STRUCT
# RUN: not kgen %s -elaborate -D TEST_FOLDER_HARDENING_FIELD=1 2>&1 | FileCheck %s --check-prefix=CHECK-FOLDER-FIELD
# RUN: not kgen %s -elaborate -D TEST_ENUMERATE_THEN_FETCH_REPEATABLE=1 2>&1 | FileCheck %s --check-prefix=CHECK-ENUMERATE

from std.sys import get_defined_bool


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


# `decorator_of[D]()` must reject a `D` conforming to `RepeatableDecorator` at
# compile time -- see `reflect.mojo`'s `decorator_of`. Deleting that
# `comptime assert` would leave every other reflection suite green; this pins
# it as a real compile-time failure of the real stdlib API.
def test_decorator_of_repeatable_struct():
    comptime if get_defined_bool[
        "TEST_DECORATOR_OF_REPEATABLE_STRUCT", False
    ]():
        # CHECK-STRUCT: decorator_of[D]() cannot be used with a decorator conforming to RepeatableDecorator
        comptime assert Bool(
            reflect[Flag].decorator_of[alias_name]()
        ), "should not reach here"


def test_decorator_of_repeatable_field():
    comptime if get_defined_bool[
        "TEST_DECORATOR_OF_REPEATABLE_FIELD", False
    ]():
        # CHECK-FIELD: decorator_of[D]() cannot be used with a decorator conforming to RepeatableDecorator
        comptime assert Bool(
            reflect[RepeatedField].member_at[0].decorator_of[alias_name]()
        ), "should not reach here"


# Defense in depth: the folder attributes backing `decorator_of`
# (`#kgen.struct_decorator_of` / `#kgen.struct_field_decorator_of`) cap their
# own result at one match, so a hand-written `__mlir_attr` spelling that
# bypasses the stdlib wrapper above -- the only other way to reach these
# attributes -- is *also* rejected, rather than silently answering with the
# first of several matches.
comptime raw_decorator_of[T: AnyType, D: AnyType] = ParameterList[
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

comptime raw_field_decorator_of[
    T: AnyType, idx: Int, D: AnyType
] = ParameterList[
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


def test_folder_hardening_struct():
    comptime if get_defined_bool["TEST_FOLDER_HARDENING_STRUCT", False]():
        # CHECK-FOLDER-STRUCT: struct_decorator_of found 2 matching decorators, but at most one is expected here
        comptime assert (
            raw_decorator_of[Flag, alias_name]().size == 1
        ), "should not reach here"


def test_folder_hardening_field():
    comptime if get_defined_bool["TEST_FOLDER_HARDENING_FIELD", False]():
        # CHECK-FOLDER-FIELD: struct_field_decorator_of found 2 matching decorators, but at most one is expected here
        comptime assert (
            raw_field_decorator_of[RepeatedField, 0, alias_name]().size == 1
        ), "should not reach here"


# Gap 5 (enumerate then fetch) x Gap 4 (repeatable decorators): a generic
# `decorator_types()` -> `decorator_of[tys[i]]()` loop hard-errors when an
# enumerated type happens to conform to `RepeatableDecorator`, rather than
# silently returning one of several matches. See the note on
# `Reflected.decorator_types`.
def test_enumerate_then_fetch_repeatable():
    comptime if get_defined_bool[
        "TEST_ENUMERATE_THEN_FETCH_REPEATABLE", False
    ]():
        comptime tys = reflect[Flag].decorator_types()
        # CHECK-ENUMERATE: decorator_of[D]() cannot be used with a decorator conforming to RepeatableDecorator
        comptime assert Bool(
            reflect[Flag].decorator_of[tys[0]]()
        ), "should not reach here"


def main():
    test_decorator_of_repeatable_struct()
    test_decorator_of_repeatable_field()
    test_folder_hardening_struct()
    test_folder_hardening_field()
    test_enumerate_then_fetch_repeatable()
