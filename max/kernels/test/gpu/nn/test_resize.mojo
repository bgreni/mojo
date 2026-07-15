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
"""Validates the GPU resize kernels (nearest / linear) against the CPU path.

The CPU implementations are exercised extensively by `test/nn/test_resize.mojo`,
so here we treat the CPU output as the reference and assert that the GPU kernels
produce the same result for every supported coordinate-transformation mode.
This closes the gap where nearest/linear resize compiled and ran only on CPU:
the GPU path silently fell back to `target="cpu"` and failed to compile on
Metal.
"""

from std.gpu.host import DeviceContext
from layout import TileTensor, row_major
from nn.resize import (
    CoordinateTransformationMode,
    RoundMode,
    resize_linear,
    resize_nearest_neighbor,
)
from std.testing import assert_almost_equal


def check_nearest[
    mode: CoordinateTransformationMode,
    round: RoundMode,
    N: Int,
    C: Int,
    IH: Int,
    IW: Int,
    OH: Int,
    OW: Int,
](gpu: DeviceContext, cpu: DeviceContext, name: StaticString) raises:
    comptime IN = N * C * IH * IW
    comptime OUT = N * C * OH * OW

    # Reference: run the CPU kernel on host memory.
    var in_host = gpu.enqueue_create_host_buffer[DType.float32](IN)
    var ref_host = gpu.enqueue_create_host_buffer[DType.float32](OUT)
    gpu.synchronize()
    var in_t = TileTensor(in_host, row_major[N, C, IH, IW]())
    var ref_t = TileTensor(ref_host, row_major[N, C, OH, OW]())
    for n in range(N):
        for c in range(C):
            for h in range(IH):
                for w in range(IW):
                    var v = ((n * C + c) * IH + h) * IW + w
                    in_t[n, c, h, w] = Float32(v) / 32.0
    resize_nearest_neighbor[mode, round, target="cpu"](in_t, ref_t, cpu)

    # Under test: run the GPU kernel on device memory.
    var in_dev = gpu.enqueue_create_buffer[DType.float32](IN)
    var out_dev = gpu.enqueue_create_buffer[DType.float32](OUT)
    var out_host = gpu.enqueue_create_host_buffer[DType.float32](OUT)
    gpu.enqueue_copy(in_dev, in_host)
    var in_dt = TileTensor(in_dev, row_major[N, C, IH, IW]())
    var out_dt = TileTensor(out_dev, row_major[N, C, OH, OW]())
    resize_nearest_neighbor[mode, round, target="gpu"](in_dt, out_dt, gpu)
    gpu.enqueue_copy(out_host, out_dev)
    gpu.synchronize()

    var out_t = TileTensor(out_host, row_major[N, C, OH, OW]())
    for n in range(N):
        for c in range(C):
            for h in range(OH):
                for w in range(OW):
                    # Nearest is a pure gather: GPU and CPU must match exactly.
                    assert_almost_equal(
                        out_t[n, c, h, w],
                        ref_t[n, c, h, w],
                        atol=0.0,
                        rtol=0.0,
                    )
    _ = in_dev^
    _ = out_dev^
    print(name, "OK")


def check_linear[
    mode: CoordinateTransformationMode,
    antialias: Bool,
    N: Int,
    CIN: Int,
    COUT: Int,
    IH: Int,
    IW: Int,
    OH: Int,
    OW: Int,
](gpu: DeviceContext, cpu: DeviceContext, name: StaticString) raises:
    comptime IN = N * CIN * IH * IW
    comptime OUT = N * COUT * OH * OW

    var in_host = gpu.enqueue_create_host_buffer[DType.float32](IN)
    var ref_host = gpu.enqueue_create_host_buffer[DType.float32](OUT)
    gpu.synchronize()
    var in_t = TileTensor(in_host, row_major[N, CIN, IH, IW]())
    var ref_t = TileTensor(ref_host, row_major[N, COUT, OH, OW]())
    for n in range(N):
        for c in range(CIN):
            for h in range(IH):
                for w in range(IW):
                    var v = ((n * CIN + c) * IH + h) * IW + w
                    in_t[n, c, h, w] = Float32(v) / 32.0
    resize_linear[mode, antialias, target="cpu"](in_t, ref_t, cpu)

    var in_dev = gpu.enqueue_create_buffer[DType.float32](IN)
    var out_dev = gpu.enqueue_create_buffer[DType.float32](OUT)
    var out_host = gpu.enqueue_create_host_buffer[DType.float32](OUT)
    gpu.enqueue_copy(in_dev, in_host)
    var in_dt = TileTensor(in_dev, row_major[N, CIN, IH, IW]())
    var out_dt = TileTensor(out_dev, row_major[N, COUT, OH, OW]())
    resize_linear[mode, antialias, target="gpu"](in_dt, out_dt, gpu)
    gpu.enqueue_copy(out_host, out_dev)
    gpu.synchronize()

    var out_t = TileTensor(out_host, row_major[N, COUT, OH, OW]())
    for n in range(N):
        for c in range(COUT):
            for h in range(OH):
                for w in range(OW):
                    assert_almost_equal(
                        out_t[n, c, h, w],
                        ref_t[n, c, h, w],
                        atol=1e-5,
                        rtol=1e-4,
                    )
    _ = in_dev^
    _ = out_dev^
    print(name, "OK")


def main() raises:
    var cpu = DeviceContext(api="cpu")
    with DeviceContext() as gpu:
        # Nearest neighbor: every coordinate-transformation / round mode,
        # upsampling and downsampling.
        check_nearest[
            CoordinateTransformationMode.HalfPixel,
            RoundMode.HalfDown,
            1,
            3,
            8,
            8,
            16,
            16,
        ](gpu, cpu, "nearest_halfpixel_upsample")
        check_nearest[
            CoordinateTransformationMode.AlignCorners,
            RoundMode.Floor,
            1,
            3,
            4,
            4,
            8,
            8,
        ](gpu, cpu, "nearest_align_corners_upsample")
        check_nearest[
            CoordinateTransformationMode.Asymmetric,
            RoundMode.HalfUp,
            1,
            2,
            5,
            5,
            9,
            9,
        ](gpu, cpu, "nearest_asymmetric_upsample")
        check_nearest[
            CoordinateTransformationMode.HalfPixel,
            RoundMode.HalfDown,
            1,
            3,
            8,
            8,
            3,
            5,
        ](gpu, cpu, "nearest_halfpixel_downsample")

        # Linear: upsampling / downsampling, every coordinate-transformation
        # mode, antialiased downsampling, and a >2 resized-dim case that
        # exercises the ping-pong device scratch buffers.
        check_linear[
            CoordinateTransformationMode.HalfPixel, False, 1, 3, 3, 8, 8, 16, 16
        ](gpu, cpu, "linear_halfpixel_upsample")
        check_linear[
            CoordinateTransformationMode.AlignCorners,
            False,
            1,
            3,
            3,
            2,
            2,
            4,
            4,
        ](gpu, cpu, "linear_align_corners_upsample")
        check_linear[
            CoordinateTransformationMode.Asymmetric,
            False,
            1,
            2,
            2,
            6,
            6,
            10,
            10,
        ](gpu, cpu, "linear_asymmetric_upsample")
        check_linear[
            CoordinateTransformationMode.HalfPixel, False, 1, 3, 3, 8, 8, 4, 3
        ](gpu, cpu, "linear_halfpixel_downsample")
        check_linear[
            CoordinateTransformationMode.HalfPixel, True, 1, 3, 3, 8, 8, 3, 4
        ](gpu, cpu, "linear_antialias_downsample")
        # Resize channel + both spatial dims: 3 resized dims -> multiple passes.
        check_linear[
            CoordinateTransformationMode.HalfPixel, False, 1, 2, 4, 3, 4, 6, 8
        ](gpu, cpu, "linear_three_resized_dims")
        # Identity: no dimension changes size -> `_resize_copy` fast path.
        check_linear[
            CoordinateTransformationMode.HalfPixel, False, 1, 3, 3, 8, 8, 8, 8
        ](gpu, cpu, "linear_no_resize")
