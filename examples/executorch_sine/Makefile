############################################################################
# apps/examples/executorch_sine/Makefile
#
# SPDX-License-Identifier: Apache-2.0
#
############################################################################

include $(APPDIR)/Make.defs

CXXEXT = .cc

PROGNAME  = $(CONFIG_EXAMPLES_EXECUTORCH_SINE_PROGNAME)
PRIORITY  = $(CONFIG_EXAMPLES_EXECUTORCH_SINE_PRIORITY)
STACKSIZE = $(CONFIG_EXAMPLES_EXECUTORCH_SINE_STACKSIZE)
MODULE    = $(CONFIG_EXAMPLES_EXECUTORCH_SINE)

MAINSRC = executorch_sine_main.cc
CSRCS = executorch_sine_math.c
CXXSRCS = executorch_sine_kernels.cc

EXECUTORCH_LIBS := \
	libexecutorch.a \
	libexecutorch_core.a \
	libextension_runner_util.a \
	libportable_ops_lib.a \
	libportable_kernels.a \
	libkernels_util_all_deps.a \
	libflatcc.a \
	libflatccrt.a

ifneq ($(CONFIG_EXECUTORCH_LINK_QUANTIZED),)
EXECUTORCH_LIBS += \
	libquantized_ops_lib.a \
	libquantized_kernels.a
endif

ifneq ($(CONFIG_EXECUTORCH_LINK_CORTEX_M),)
EXECUTORCH_LIBS += \
	libcortex_m_ops_lib.a \
	libcortex_m_kernels.a \
	libcmsis-nn.a
endif

ifneq ($(CONFIG_EXECUTORCH_LINK_ETHOS_U),)
EXECUTORCH_LIBS += libexecutorch_delegate_ethos_u.a
endif

EXECUTORCH_LIB_TARGETS := $(addprefix lib/,$(EXECUTORCH_LIBS))

context:: $(EXECUTORCH_LIB_TARGETS)

$(EXECUTORCH_LIB_TARGETS):
	$(Q) echo "Missing ExecuTorch library: $@"
	$(Q) echo "Run ./build.sh executorch, then stage the generated .a files under $(CURDIR)/lib"
	$(Q) exit 1

include $(APPDIR)/Application.mk
