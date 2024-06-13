# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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
# RUN: %mojo %s

from testing import assert_equal
from utils.numerics import nan


def test_abs():
    assert_equal(0, abs(0))
    assert_equal(1, abs(1))
    assert_equal(1, abs(-1))

    var lhs = SIMD[DType.int32, 4](1, -2, 3, -4)
    var expected = SIMD[DType.int32, 4](1, 2, 3, 4)
    assert_equal(expected, abs(lhs))


def test_divmod():
    var t = divmod(0, 1)
    assert_equal(0, t[0])
    assert_equal(0, t[1])
    t = divmod(1, 1)
    assert_equal(1, t[0])
    assert_equal(0, t[1])
    t = divmod(1, 2)
    assert_equal(0, t[0])
    assert_equal(1, t[1])
    t = divmod(4, 3)
    assert_equal(1, t[0])
    assert_equal(1, t[1])


def test_min():
    assert_equal(-2, min(-2, -1))
    assert_equal(-1, min(0, -1))
    assert_equal(0, min(0, 1))
    assert_equal(1, min(1, 42))

    assert_equal(UInt(0), min(UInt(0), UInt(1)))
    assert_equal(UInt(1), min(UInt(1), UInt(42)))

    var lhs = SIMD[DType.int32, 4](1, 2, 3, 4)
    var rhs = SIMD[DType.int32, 4](0, 1, 5, 7)
    var expected = SIMD[DType.int32, 4](0, 1, 3, 4)
    assert_equal(expected, lhs.min(rhs))
    assert_equal(expected, rhs.min(lhs))


def test_max():
    assert_equal(-1, max(-2, -1))
    assert_equal(0, max(0, -1))
    assert_equal(1, max(0, 1))
    assert_equal(2, max(2, 1))

    assert_equal(UInt(1), max(UInt(0), UInt(1)))
    assert_equal(UInt(2), max(UInt(1), UInt(2)))

    var lhs = SIMD[DType.int32, 4](1, 2, 3, 4)
    var rhs = SIMD[DType.int32, 4](0, 1, 5, 7)
    var expected = SIMD[DType.int32, 4](1, 2, 5, 7)
    assert_equal(expected, lhs.max(rhs))
    assert_equal(expected, rhs.max(lhs))


def test_round():
    assert_equal(0, round(0.0))
    assert_equal(1, round(1.0))
    assert_equal(1, round(1.1))
    assert_equal(1, round(1.4))
    assert_equal(2, round(1.5))
    assert_equal(2, round(2.0))
    assert_equal(1, round(1.4, 0))
    # FIXME: round(2.5) is 2.0 (roundeven) but it's using roundhalfup
    # Fix when the math libray is open sourced
    # assert_equal(2, round(2.5))
    # assert_equal(1.5, round(1.5, 1))
    # assert_equal(1.61, round(1.613, 2))

    var lhs = SIMD[DType.float32, 4](1.1, 1.5, 1.9, 2.0)
    var expected = SIMD[DType.float32, 4](1.0, 2.0, 2.0, 2.0)
    assert_equal(expected, round(lhs))

    # Ensure that round works on float literal
    alias r1: FloatLiteral = round(2.3)
    assert_equal(r1, 2.0)
    alias r2: FloatLiteral = round(2.3324, 2)
    assert_equal(r2, 2.33)


def test_pow():
    alias F = SIMD[DType.float32, 4]
    var base = F(0.0, 1.0, 2.0, 3.0)
    assert_equal(pow(base, 2.0), F(0.0, 1.0, 4.0, 9.0))
    assert_equal(pow(base, int(2)), F(0.0, 1.0, 4.0, 9.0))
    alias I = SIMD[DType.int32, 4]
    assert_equal(pow(I(0, 1, 2, 3), int(2)), I(0, 1, 4, 9))


def test_clamp():
    assert_equal(clamp(4, 5, 7), 5)
    assert_equal(clamp(9, 5, 7), 7)
    assert_equal(clamp(6, 5, 7), 6)

    var lower = Int32(3)
    var upper = Int32(9)
    assert_equal(clamp(Int32(1), lower, upper), lower)
    assert_equal(clamp(Int32(12), lower, upper), upper)
    assert_equal(clamp(Int32(7), lower, upper), 7)
    assert_equal(clamp(Int32(1), upper, lower), lower)

    assert_equal(clamp(nan[DType.float64](), 1.3, 5.4), 5.4)

    var value = SIMD[DType.int32, 4](4, 6, 65, 1)
    assert_equal(
        clamp(value, Int32(3), Int32(8)), SIMD[DType.int32, 4](4, 6, 8, 3)
    )

    assert_equal(clamp(2, 6, 4), 4)
    assert_equal(clamp(9, 6, 4), 6)


def main():
    test_abs()
    test_divmod()
    test_max()
    test_min()
    test_round()
    test_pow()
    test_clamp()
