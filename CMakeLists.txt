cmake_minimum_required(VERSION 3.16)
project(ndl-toolchain CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)

# --- libndl_rt (runtime, statically linked into every NDL binary) -----------
add_library(ndl_rt STATIC
  runtime/ndl_rt.cpp
  runtime/ndl_rt_network.cpp
  runtime/ndl_rt_io.cpp
  runtime/ndl_rt_gpu.cpp
  runtime/ndl_rt_backend.cpp
  runtime/ndl_rt_ocl.cpp
  runtime/ndl_rt_sched.cpp
  runtime/ndl_rt_v2.cpp)
target_include_directories(ndl_rt PUBLIC runtime)
# Windows (MinGW): no libdl / libm. winpthreads provides std::thread.
if(WIN32)
  target_link_libraries(ndl_rt PUBLIC Threads::Threads)
else()
  target_link_libraries(ndl_rt PUBLIC Threads::Threads ${CMAKE_DL_LIBS} m)
endif()

# --- ndlc (compiler driver) --------------------------------------------------
add_executable(ndlc
  src/main.cpp src/diag.cpp src/lexer.cpp src/parser.cpp src/sema.cpp
  src/irgen_llvm.cpp src/irgen_ptx.cpp src/vm.cpp src/toml.cpp
  src/manifest.cpp src/linker.cpp)
target_include_directories(ndlc PRIVATE include runtime)
target_link_libraries(ndlc PRIVATE ndl_rt)

install(TARGETS ndlc RUNTIME DESTINATION bin)
install(TARGETS ndl_rt ARCHIVE DESTINATION lib)
