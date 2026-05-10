/****************************************************************************
 * apps/examples/executorch_sine/executorch_sine_main.cc
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 ****************************************************************************/

#include <nuttx/config.h>

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <utility>

#include <executorch/runtime/core/data_loader.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/freeable_buffer.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/executor/memory_manager.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/method_meta.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/runtime.h>

#include "sine_wave_portable_pte.h"

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::aten::TensorImpl;
using executorch::runtime::DataLoader;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::MemoryManager;
using executorch::runtime::Method;
using executorch::runtime::MethodMeta;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::Span;
using executorch::runtime::Tag;
using executorch::runtime::TensorInfo;

extern "C" Error executorch_sine_register_kernels(void);

namespace {

constexpr size_t kMaxDims = 4;
constexpr size_t kMaxInputFloats = 16;
constexpr size_t kMaxOutputPrintFloats = 16;
constexpr float kDefaultStart = 0.0f;
constexpr float kDefaultStep = 0.1f;
constexpr unsigned int kDefaultDelayMs = 100;
constexpr int kRunForever = 0;
constexpr float kPi = 3.14159265358979323846f;
constexpr float kTwoPi = 6.28318530717958647692f;

static uint8_t
    g_method_allocator_pool[CONFIG_EXAMPLES_EXECUTORCH_SINE_METHOD_ALLOCATOR_SIZE];
static uint8_t
    g_planned_buffer[CONFIG_EXAMPLES_EXECUTORCH_SINE_PLANNED_BUFFER_SIZE];
static float g_input_data[kMaxInputFloats];
static TensorImpl::SizesType g_input_sizes[kMaxDims];
static TensorImpl::DimOrderType g_input_dim_order[kMaxDims];

class EmbeddedBufferDataLoader final : public DataLoader
{
 public:
  EmbeddedBufferDataLoader(const void *data, size_t size)
      : data_(static_cast<const uint8_t *>(data)), size_(size)
  {
  }

  Result<executorch::runtime::FreeableBuffer> load(
      size_t offset,
      size_t size,
      const DataLoader::SegmentInfo &segment_info) const override
  {
    (void)segment_info;
    if (offset > size_ || size > size_ - offset)
      {
        return Error::InvalidArgument;
      }

    return executorch::runtime::FreeableBuffer(
        data_ + offset, size, nullptr);
  }

  Error load_into(
      size_t offset,
      size_t size,
      const SegmentInfo &segment_info,
      void *buffer) const override
  {
    (void)segment_info;
    if (buffer == nullptr || offset > size_ || size > size_ - offset)
      {
        return Error::InvalidArgument;
      }

    memcpy(buffer, data_ + offset, size);
    return Error::Ok;
  }

  Result<size_t> size() const override
  {
    return size_;
  }

 private:
  const uint8_t *data_;
  size_t size_;
};

uint64_t monotonic_us()
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<uint64_t>(ts.tv_sec) * 1000000ull +
         static_cast<uint64_t>(ts.tv_nsec / 1000);
}

const char *tag_name(Tag tag)
{
  switch (tag)
    {
      case Tag::None:
        return "None";
      case Tag::Tensor:
        return "Tensor";
      case Tag::String:
        return "String";
      case Tag::Double:
        return "Double";
      case Tag::Int:
        return "Int";
      case Tag::Bool:
        return "Bool";
      default:
        return "Other";
    }
}

void print_tensor_info(const char *label, const TensorInfo &info)
{
  printf("%s: type=%d shape=[", label,
         static_cast<int>(info.scalar_type()));
  Span<const int32_t> sizes = info.sizes();
  for (size_t i = 0; i < sizes.size(); ++i)
    {
      printf("%s%" PRId32, i == 0 ? "" : ", ", sizes[i]);
    }

  printf("] nbytes=%zu planned=%s\n",
         info.nbytes(), info.is_memory_planned() ? "yes" : "no");
}

size_t tensor_numel(const TensorInfo &info)
{
  size_t numel = 1;
  Span<const int32_t> sizes = info.sizes();
  for (size_t i = 0; i < sizes.size(); ++i)
    {
      if (sizes[i] <= 0)
        {
          return 0;
        }

      numel *= static_cast<size_t>(sizes[i]);
    }

  return numel;
}

bool setup_input_tensor(
    const TensorInfo &input_info,
    float value,
    EValue *input_evalue)
{
  if (input_info.scalar_type() != ScalarType::Float)
    {
      printf("Only Float input tensors are supported by this sample.\n");
      return false;
    }

  const size_t ndim = input_info.sizes().size();
  if (ndim == 0 || ndim > kMaxDims)
    {
      printf("Unsupported input rank: %zu\n", ndim);
      return false;
    }

  const size_t numel = tensor_numel(input_info);
  if (numel == 0 || numel > kMaxInputFloats)
    {
      printf("Unsupported input element count: %zu\n", numel);
      return false;
    }

  for (size_t i = 0; i < ndim; ++i)
    {
      g_input_sizes[i] =
          static_cast<TensorImpl::SizesType>(input_info.sizes()[i]);
      g_input_dim_order[i] = static_cast<TensorImpl::DimOrderType>(i);
    }

  for (size_t i = 0; i < numel; ++i)
    {
      g_input_data[i] = value;
    }

  static TensorImpl input_impl(
      ScalarType::Float,
      static_cast<ssize_t>(ndim),
      g_input_sizes,
      g_input_data,
      g_input_dim_order);
  Tensor input_tensor(&input_impl);
  *input_evalue = EValue(input_tensor);
  return true;
}

bool print_output(const EValue &output)
{
  if (output.isTensor())
    {
      const Tensor &tensor = output.toTensor();
      if (tensor.scalar_type() != ScalarType::Float)
        {
          printf("Output tensor type=%d is not Float\n",
                 static_cast<int>(tensor.scalar_type()));
          return false;
        }

      const float *data = tensor.const_data_ptr<float>();
      const ssize_t count = tensor.numel();
      const size_t print_count =
          count < static_cast<ssize_t>(kMaxOutputPrintFloats) ?
          static_cast<size_t>(count) : kMaxOutputPrintFloats;

      printf("Output tensor shape=[");
      auto sizes = tensor.sizes();
      for (size_t i = 0; i < sizes.size(); ++i)
        {
          printf("%s%" PRId32, i == 0 ? "" : ", ", sizes[i]);
        }

      printf("] values=");
      for (size_t i = 0; i < print_count; ++i)
        {
          printf("%s%.6f", i == 0 ? "[" : ", ", data[i]);
        }

      printf("%s\n", count > static_cast<ssize_t>(print_count) ? ", ...]" : "]");
      return true;
    }

  if (output.isDouble())
    {
      printf("Output double=%.6f\n", output.toDouble());
      return true;
    }

  if (output.isInt())
    {
      printf("Output int=%" PRId64 "\n", output.toInt());
      return true;
    }

  printf("Unsupported output tag: %s\n", tag_name(output.tag));
  return false;
}

bool read_first_output_float(const EValue &output, float *value)
{
  if (output.isTensor())
    {
      const Tensor &tensor = output.toTensor();
      if (tensor.scalar_type() != ScalarType::Float || tensor.numel() < 1)
        {
          return false;
        }

      *value = tensor.const_data_ptr<float>()[0];
      return true;
    }

  if (output.isDouble())
    {
      *value = static_cast<float>(output.toDouble());
      return true;
    }

  if (output.isInt())
    {
      *value = static_cast<float>(output.toInt());
      return true;
    }

  return false;
}

void print_fixed6(float value)
{
  int64_t scaled = static_cast<int64_t>(
      value * 1000000.0f + (value >= 0.0f ? 0.5f : -0.5f));

  if (scaled < 0)
    {
      printf("-");
      scaled = -scaled;
    }

  printf("%" PRId64 ".%06" PRId64, scaled / 1000000, scaled % 1000000);
}

float wrap_phase(float phase)
{
  while (phase > kPi)
    {
      phase -= kTwoPi;
    }

  while (phase < -kPi)
    {
      phase += kTwoPi;
    }

  return phase;
}

float sine_approx(float phase)
{
  const float x = wrap_phase(phase);
  const float x2 = x * x;

  return x * (1.0f - x2 * (1.0f / 6.0f) +
              x2 * x2 * (1.0f / 120.0f) -
              x2 * x2 * x2 * (1.0f / 5040.0f));
}

} // namespace

extern "C" int main(int argc, char *argv[])
{
  const float start_value =
      argc > 1 ? static_cast<float>(atof(argv[1])) : kDefaultStart;
  const float step =
      argc > 2 ? static_cast<float>(atof(argv[2])) : kDefaultStep;
  const unsigned int delay_ms =
      argc > 3 ? static_cast<unsigned int>(atoi(argv[3])) : kDefaultDelayMs;
  const int sample_count =
      argc > 4 ? atoi(argv[4]) : kRunForever;

  printf("ExecuTorch sine example\n");
  printf("PTE size: %u bytes\n", executorch_export_sine_wave_portable_pte_len);
  printf("Start: ");
  print_fixed6(start_value);
  printf(" Step: ");
  print_fixed6(step);
  printf(" Delay: %u ms Count: %s",
         delay_ms,
         sample_count == kRunForever ? "forever" : "");
  if (sample_count != kRunForever)
    {
      printf("%d", sample_count);
    }

  printf("\n");

  executorch::runtime::runtime_init();
  Error err = executorch_sine_register_kernels();
  if (err != Error::Ok)
    {
      printf("register_kernels failed: %s (%d)\n",
             executorch::runtime::to_string(err), static_cast<int>(err));
      return 1;
    }

  EmbeddedBufferDataLoader loader(
      executorch_export_sine_wave_portable_pte,
      executorch_export_sine_wave_portable_pte_len);

  Result<Program> program_result = Program::load(&loader);
  if (!program_result.ok())
    {
      err = program_result.error();
      printf("Program::load failed: %s (%d)\n",
             executorch::runtime::to_string(err), static_cast<int>(err));
      return 1;
    }

  Program program = std::move(*program_result);
  printf("Program loaded. methods=%zu\n", program.num_methods());

  Result<const char *> method_name_result = program.get_method_name(0);
  if (!method_name_result.ok())
    {
      printf("get_method_name failed: %d\n",
             static_cast<int>(method_name_result.error()));
      return 1;
    }

  const char *method_name = *method_name_result;
  printf("Using method: %s\n", method_name);

  Result<MethodMeta> method_meta_result = program.method_meta(method_name);
  if (!method_meta_result.ok())
    {
      printf("method_meta failed: %d\n",
             static_cast<int>(method_meta_result.error()));
      return 1;
    }

  MethodMeta method_meta = std::move(*method_meta_result);
  printf("Inputs=%zu Outputs=%zu Planned buffers=%zu\n",
         method_meta.num_inputs(),
         method_meta.num_outputs(),
         method_meta.num_memory_planned_buffers());

  if (method_meta.num_inputs() != 1 || method_meta.num_outputs() < 1)
    {
      printf("This sample expects one input and at least one output.\n");
      return 1;
    }

  Result<Tag> input_tag = method_meta.input_tag(0);
  if (!input_tag.ok() || *input_tag != Tag::Tensor)
    {
      printf("Input 0 tag is %s, expected Tensor\n",
             input_tag.ok() ? tag_name(*input_tag) : "Error");
      return 1;
    }

  Result<TensorInfo> input_info_result = method_meta.input_tensor_meta(0);
  if (!input_info_result.ok())
    {
      printf("input_tensor_meta failed: %d\n",
             static_cast<int>(input_info_result.error()));
      return 1;
    }

  TensorInfo input_info = *input_info_result;
  print_tensor_info("Input 0", input_info);

  for (size_t i = 0; i < method_meta.num_outputs(); ++i)
    {
      Result<Tag> output_tag = method_meta.output_tag(i);
      printf("Output %zu tag=%s\n", i,
             output_tag.ok() ? tag_name(*output_tag) : "Error");
      if (output_tag.ok() && *output_tag == Tag::Tensor)
        {
          Result<TensorInfo> output_info = method_meta.output_tensor_meta(i);
          if (output_info.ok())
            {
              print_tensor_info("Output tensor", *output_info);
            }
        }
    }

  if (method_meta.num_memory_planned_buffers() > 1)
    {
      printf("This sample supports one planned memory buffer.\n");
      return 1;
    }

  Span<uint8_t> planned_spans[1] = {{g_planned_buffer, sizeof(g_planned_buffer)}};
  if (method_meta.num_memory_planned_buffers() == 1)
    {
      Result<int64_t> required_size = method_meta.memory_planned_buffer_size(0);
      if (!required_size.ok())
        {
          printf("memory_planned_buffer_size failed: %d\n",
                 static_cast<int>(required_size.error()));
          return 1;
        }

      printf("Planned buffer 0: required=%" PRId64 " available=%zu\n",
             *required_size, sizeof(g_planned_buffer));
      if (*required_size > static_cast<int64_t>(sizeof(g_planned_buffer)))
        {
          printf("Increase CONFIG_EXAMPLES_EXECUTORCH_SINE_PLANNED_BUFFER_SIZE.\n");
          return 1;
        }
    }

  MemoryAllocator method_allocator(
      sizeof(g_method_allocator_pool),
      g_method_allocator_pool);
  HierarchicalAllocator planned_memory(
      {planned_spans, method_meta.num_memory_planned_buffers()});
  MemoryManager memory_manager(&method_allocator, &planned_memory);

  Result<Method> method_result =
      program.load_method(method_name, &memory_manager);
  if (!method_result.ok())
    {
      err = method_result.error();
      printf("load_method failed: %s (%d)\n",
             executorch::runtime::to_string(err), static_cast<int>(err));
      return 1;
    }

  Method method = std::move(*method_result);
  EValue input;
  if (!setup_input_tensor(input_info, start_value, &input))
    {
      return 1;
    }

  err = method.set_input(input, 0);
  if (err != Error::Ok)
    {
      printf("set_input failed: %d\n", static_cast<int>(err));
      return 1;
    }

  printf("sample,phase,sine,model_elapsed_us\n");

  float input_value = start_value;
  for (int sample = 0;
       sample_count == kRunForever || sample < sample_count;
       ++sample)
    {
      for (size_t i = 0; i < tensor_numel(input_info); ++i)
        {
          g_input_data[i] = input_value;
        }

      const uint64_t start = monotonic_us();
      err = method.execute();
      const uint64_t elapsed = monotonic_us() - start;
      if (err != Error::Ok)
        {
          printf("execute failed: %d\n", static_cast<int>(err));
          return 1;
        }

      printf("%d,", sample);
      print_fixed6(input_value);
      printf(",");
      print_fixed6(sine_approx(input_value));
      printf(",%" PRIu64 "\n", elapsed);
      fflush(stdout);

      input_value += step;
      if (input_value >= kTwoPi || input_value <= -kTwoPi)
        {
          input_value = wrap_phase(input_value);
        }

      if (delay_ms > 0)
        {
          usleep(delay_ms * 1000);
        }
    }

  return 0;
}
