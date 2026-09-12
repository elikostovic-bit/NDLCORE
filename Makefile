# NDL Toolchain build (primary build system)
# Targets: dist/bin/ndlc + dist/lib/libndl_rt.a + dist/std/
# Usage: make -j$(nproc)          (add VERBOSE=1 for command echo)

CXX      ?= g++
CXXSTD   := -std=c++20
OPT      ?= -O2
WARN     := -Wall -Wextra
INCLUDES := -Iinclude -Iruntime
# Windows (MinGW): no libdl; winpthreads provides the std::thread symbols.
# Override if your MinGW lacks libwinpthread.a:  make LIBS=
ifeq ($(OS),Windows_NT)
LIBS     ?= -lwinpthread
else
LIBS     ?= -lpthread -ldl -lm
endif

BUILD    := build
DIST     := dist

COMPILER_SRCS := src/main.cpp src/diag.cpp src/lexer.cpp src/parser.cpp src/sema.cpp \
	         src/irgen_llvm.cpp src/irgen_ptx.cpp src/vm.cpp src/toml.cpp \
	         src/manifest.cpp src/linker.cpp
RUNTIME_SRCS  := runtime/ndl_rt.cpp runtime/ndl_rt_network.cpp runtime/ndl_rt_io.cpp \
	         runtime/ndl_rt_gpu.cpp runtime/ndl_rt_backend.cpp runtime/ndl_rt_ocl.cpp \
	         runtime/ndl_rt_sched.cpp runtime/ndl_rt_v2.cpp

COMPILER_OBJS := $(patsubst src/%.cpp,$(BUILD)/obj/%.o,$(COMPILER_SRCS))
RUNTIME_OBJS  := $(patsubst runtime/%.cpp,$(BUILD)/rtobj/%.o,$(RUNTIME_SRCS))

.PHONY: all clean test
all: $(DIST)/bin/ndlc $(DIST)/lib/libndl_rt.a $(DIST)/std/stdlib/math.ndl

$(BUILD)/obj/%.o: src/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXSTD) $(OPT) $(WARN) $(INCLUDES) -c $< -o $@

$(BUILD)/rtobj/%.o: runtime/%.cpp runtime/ndl_rt.h runtime/ndl_rt_internal.h
	@mkdir -p $(@D)
	$(CXX) $(CXXSTD) $(OPT) $(WARN) -Iruntime -c $< -o $@

$(DIST)/lib/libndl_rt.a: $(RUNTIME_OBJS)
	@mkdir -p $(dir $@)
	ar rcs $@ $(RUNTIME_OBJS)

$(DIST)/bin/ndlc: $(COMPILER_OBJS) $(DIST)/lib/libndl_rt.a
	@mkdir -p $(dir $@)
	$(CXX) $(CXXSTD) $(OPT) $(COMPILER_OBJS) $(DIST)/lib/libndl_rt.a $(LIBS) -o $@

$(DIST)/std/stdlib/math.ndl: stdlib/math.ndl
	@mkdir -p $(dir $@)
	cp $< $@

clean:
	rm -rf $(BUILD) $(DIST)/bin $(DIST)/lib

test: all
	$(DIST)/bin/ndlc check main.ndl
	$(DIST)/bin/ndlc check examples/minimal.ndl
	$(DIST)/bin/ndlc check examples/stdp_demo.ndl
	$(DIST)/bin/ndlc check examples/v2_demo.ndl
	$(DIST)/bin/ndlc check agi/agi.ndl
