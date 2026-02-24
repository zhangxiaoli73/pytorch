# ---[ xpu

# Poor man's include guard
if(TARGET torch::xpurt)
  return()
endif()

set(XPU_HOST_CXX_FLAGS)

# Find SYCL library.
find_package(SYCLToolkit REQUIRED)
if(NOT SYCL_FOUND)
  set(PYTORCH_FOUND_XPU FALSE)
  # Exit early to avoid populating XPU_HOST_CXX_FLAGS.
  return()
endif()
# Find LevelZero library.
if(USE_LEVEL_ZERO)
  find_package(LevelZero REQUIRED)
  if(NOT LevelZero_FOUND)
    set(PYTORCH_FOUND_XPU FALSE)
    return()
  endif()
endif()
set(PYTORCH_FOUND_XPU TRUE)

# SYCL library interface
add_library(torch::sycl INTERFACE IMPORTED)

set_property(
    TARGET torch::sycl PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    ${SYCL_INCLUDE_DIR})
set_property(
    TARGET torch::sycl PROPERTY INTERFACE_LINK_LIBRARIES
    ${SYCL_LIBRARY})

# LevelZero library interface
if(USE_LEVEL_ZERO)
  add_library(torch::level_zero INTERFACE IMPORTED)

  set_property(
      TARGET torch::level_zero PROPERTY INTERFACE_INCLUDE_DIRECTORIES
      ${LevelZero_INCLUDE_DIR})
  set_property(
      TARGET torch::level_zero PROPERTY INTERFACE_LINK_LIBRARIES
      ${LevelZero_LIBRARY})
endif()

# xpurt
add_library(torch::xpurt INTERFACE IMPORTED)

set(xpurt_deps torch::sycl)
if(USE_LEVEL_ZERO)
  list(APPEND xpurt_deps torch::level_zero)
endif()

set_property(
    TARGET torch::xpurt PROPERTY INTERFACE_LINK_LIBRARIES
    "${xpurt_deps}")

# setting xpu arch flags
torch_xpu_get_arch_list(XPU_ARCH_FLAGS)
# propagate to torch-xpu-ops
set(TORCH_XPU_ARCH_LIST ${XPU_ARCH_FLAGS})

# Ensure USE_XPU is enabled.
string(APPEND XPU_HOST_CXX_FLAGS " -DUSE_XPU")
string(APPEND XPU_HOST_CXX_FLAGS " -DSYCL_COMPILER_VERSION=${SYCL_COMPILER_VERSION}")

if(DEFINED ENV{XPU_ENABLE_KINETO})
  set(XPU_ENABLE_KINETO TRUE)
else()
  set(XPU_ENABLE_KINETO FALSE)
endif()

if(WIN32)
  if(${SYCL_COMPILER_VERSION} GREATER_EQUAL 20250101)
    set(XPU_ENABLE_KINETO TRUE)
  endif()
else()
  set(XPU_ENABLE_KINETO TRUE)
endif()