# This will define the following variables:
# LevelZero_FOUND          : True if the system has the LevelZero library.
# LevelZero_INCLUDE_DIR    : Level Zero include directory.
# LevelZero_LIBRARY        : Level Zero library directory.

include(FindPackageHandleStandardArgs)

# Find Level Zero include directory.
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
  find_path(
    LevelZero_INCLUDE_DIR
    NAMES level_zero/ze_api.h
    PATH_SUFFIXES include
    )
elseif(CMAKE_SYSTEM_NAME MATCHES "Windows")
  find_path(
    LevelZero_INCLUDE_DIR
    NAMES level_zero/ze_api.h
    HINTS $ENV{LEVEL_ZERO_V1_SDK_PATH}
    PATH_SUFFIXES include
    )
endif()

if(NOT LevelZero_INCLUDE_DIR)
  set(LevelZero_FOUND False)
  set(LevelZero_REASON_FAILURE "Level Zero include directory not found!!")
  set(LevelZero_NOT_FOUND_MESSAGE "${LevelZero_REASON_FAILURE}")
  return()
endif()

# Find Level Zero library directory.
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
  find_library(
    LevelZero_LIBRARY
    NAMES ze_loader
    PATH_SUFFIXES x86_64_linux_gnu lib lib/x64 lib64
    )
elseif(CMAKE_SYSTEM_NAME MATCHES "Windows")
  find_library(
    LevelZero_LIBRARY
    NAMES ze_loader
    HINTS $ENV{LEVEL_ZERO_V1_SDK_PATH}
    PATH_SUFFIXES lib lib64
    )
endif()

if(NOT LevelZero_LIBRARY)
  set(LevelZero_FOUND False)
  set(LevelZero_REASON_FAILURE "Level Zero library not found!!")
  set(LevelZero_NOT_FOUND_MESSAGE "${SYCL_REASON_FAILURE}")
  return()
endif()

find_package_handle_standard_args(
  LevelZero
  FOUND_VAR LevelZero_FOUND
  REQUIRED_VARS LevelZero_INCLUDE_DIR LevelZero_LIBRARY
  REASON_FAILURE_MESSAGE "${LevelZero_REASON_FAILURE}"
  )
