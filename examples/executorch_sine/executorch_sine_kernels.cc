/****************************************************************************
 * apps/examples/executorch_sine/executorch_sine_kernels.cc
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 ****************************************************************************/

#include <optional>

#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/kernel/operator_registry.h>

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::ArrayRef;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::Kernel;
using executorch::runtime::KernelRuntimeContext;
using executorch::runtime::Span;

namespace cortex_m {
namespace native {

Tensor &quantize_per_tensor_out(KernelRuntimeContext &context,
                                const Tensor &input, double scale,
                                int64_t zero_point, int64_t quant_min,
                                int64_t quant_max, ScalarType dtype,
                                Tensor &out);

Tensor &dequantize_per_tensor_out(KernelRuntimeContext &context,
                                  const Tensor &input, double scale,
                                  int64_t zero_point, int64_t quant_min,
                                  int64_t quant_max, ScalarType dtype,
                                  Tensor &out);

Tensor &quantized_linear_out(KernelRuntimeContext &context,
                             const Tensor &input, const Tensor &weights,
                             const std::optional<Tensor> &bias,
                             const std::optional<Tensor> &kernel_sum,
                             int64_t input_offset, int64_t filter_offset,
                             int64_t output_offset,
                             ArrayRef<int64_t> requantize_multipliers,
                             ArrayRef<int64_t> requantize_shifts,
                             int64_t activation_max,
                             int64_t activation_min, Tensor &out);

Tensor &maximum_out(KernelRuntimeContext &context, const Tensor &self,
                    const Tensor &other, Tensor &out);

Tensor &minimum_out(KernelRuntimeContext &context, const Tensor &self,
                    const Tensor &other, Tensor &out);

} // namespace native
} // namespace cortex_m

namespace torch {
namespace executor {
namespace native {

Tensor &addmm_out(KernelRuntimeContext &context, const Tensor &self,
                  const Tensor &mat1, const Tensor &mat2,
                  const executorch::aten::Scalar &beta,
                  const executorch::aten::Scalar &alpha, Tensor &out);

Tensor &permute_copy_out(KernelRuntimeContext &context, const Tensor &self,
                         ArrayRef<int64_t> dims, Tensor &out);

Tensor &relu_out(KernelRuntimeContext &context, const Tensor &self,
                 Tensor &out);

} // namespace native
} // namespace executor
} // namespace torch

namespace {

void quantize_per_tensor_kernel(KernelRuntimeContext &context,
                                Span<EValue *> stack)
{
  const Tensor &input = stack[0]->to<Tensor>();
  const double scale = stack[1]->to<double>();
  const int64_t zero_point = stack[2]->to<int64_t>();
  const int64_t quant_min = stack[3]->to<int64_t>();
  const int64_t quant_max = stack[4]->to<int64_t>();
  const ScalarType dtype = stack[5]->to<ScalarType>();
  Tensor &out = stack[6]->to<Tensor>();

  cortex_m::native::quantize_per_tensor_out(
      context, input, scale, zero_point, quant_min, quant_max, dtype, out);
}

void dequantize_per_tensor_kernel(KernelRuntimeContext &context,
                                  Span<EValue *> stack)
{
  const Tensor &input = stack[0]->to<Tensor>();
  const double scale = stack[1]->to<double>();
  const int64_t zero_point = stack[2]->to<int64_t>();
  const int64_t quant_min = stack[3]->to<int64_t>();
  const int64_t quant_max = stack[4]->to<int64_t>();
  const ScalarType dtype = stack[5]->to<ScalarType>();
  Tensor &out = stack[6]->to<Tensor>();

  cortex_m::native::dequantize_per_tensor_out(
      context, input, scale, zero_point, quant_min, quant_max, dtype, out);
}

void quantized_linear_kernel(KernelRuntimeContext &context,
                             Span<EValue *> stack)
{
  const Tensor &input = stack[0]->to<Tensor>();
  const Tensor &weights = stack[1]->to<Tensor>();
  std::optional<Tensor> bias = stack[2]->toOptional<Tensor>();
  std::optional<Tensor> kernel_sum = stack[3]->toOptional<Tensor>();
  const int64_t input_offset = stack[4]->to<int64_t>();
  const int64_t filter_offset = stack[5]->to<int64_t>();
  const int64_t output_offset = stack[6]->to<int64_t>();
  ArrayRef<int64_t> requantize_multipliers = stack[7]->toIntList();
  ArrayRef<int64_t> requantize_shifts = stack[8]->toIntList();
  const int64_t activation_max = stack[9]->to<int64_t>();
  const int64_t activation_min = stack[10]->to<int64_t>();
  Tensor &out = stack[11]->to<Tensor>();

  cortex_m::native::quantized_linear_out(
      context, input, weights, bias, kernel_sum, input_offset, filter_offset,
      output_offset, requantize_multipliers, requantize_shifts, activation_max,
      activation_min, out);
}

void maximum_kernel(KernelRuntimeContext &context, Span<EValue *> stack)
{
  const Tensor &self = stack[0]->to<Tensor>();
  const Tensor &other = stack[1]->to<Tensor>();
  Tensor &out = stack[2]->to<Tensor>();

  cortex_m::native::maximum_out(context, self, other, out);
}

void minimum_kernel(KernelRuntimeContext &context, Span<EValue *> stack)
{
  const Tensor &self = stack[0]->to<Tensor>();
  const Tensor &other = stack[1]->to<Tensor>();
  Tensor &out = stack[2]->to<Tensor>();

  cortex_m::native::minimum_out(context, self, other, out);
}

void addmm_kernel(KernelRuntimeContext &context, Span<EValue *> stack)
{
  const Tensor &self = stack[0]->to<Tensor>();
  const Tensor &mat1 = stack[1]->to<Tensor>();
  const Tensor &mat2 = stack[2]->to<Tensor>();
  const executorch::aten::Scalar &beta =
      stack[3]->to<executorch::aten::Scalar>();
  const executorch::aten::Scalar &alpha =
      stack[4]->to<executorch::aten::Scalar>();
  Tensor &out = stack[5]->to<Tensor>();

  torch::executor::native::addmm_out(
      context, self, mat1, mat2, beta, alpha, out);
}

void permute_copy_kernel(KernelRuntimeContext &context, Span<EValue *> stack)
{
  const Tensor &self = stack[0]->to<Tensor>();
  ArrayRef<int64_t> dims = stack[1]->toIntList();
  Tensor &out = stack[2]->to<Tensor>();

  torch::executor::native::permute_copy_out(context, self, dims, out);
}

void relu_kernel(KernelRuntimeContext &context, Span<EValue *> stack)
{
  const Tensor &self = stack[0]->to<Tensor>();
  Tensor &out = stack[1]->to<Tensor>();

  torch::executor::native::relu_out(context, self, out);
}

} // namespace

extern "C" Error executorch_sine_register_kernels(void)
{
  static const Kernel kernels[] =
  {
    Kernel("cortex_m::quantize_per_tensor.out",
           quantize_per_tensor_kernel),
    Kernel("cortex_m::dequantize_per_tensor.out",
           dequantize_per_tensor_kernel),
    Kernel("cortex_m::quantized_linear.out", quantized_linear_kernel),
    Kernel("cortex_m::maximum.out", maximum_kernel),
    Kernel("cortex_m::minimum.out", minimum_kernel),
    Kernel("aten::addmm.out", addmm_kernel),
    Kernel("aten::permute_copy.out", permute_copy_kernel),
    Kernel("aten::relu.out", relu_kernel),
  };

  return executorch::runtime::register_kernels(
      {kernels, sizeof(kernels) / sizeof(kernels[0])});
}
