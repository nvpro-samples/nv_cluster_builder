# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

find_package(GTest QUIET)
if(NOT GTest_FOUND)
  include(FetchContent)
  FetchContent_Declare(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG v1.14.0
      GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(googletest)
endif()

add_executable(nv_cluster_builder_tests
  src/test_clusterizer.cpp
)
target_include_directories(nv_cluster_builder_tests PRIVATE src)
target_compile_definitions(nv_cluster_builder_tests PRIVATE GLM_FORCE_XYZW_ONLY GLM_ENABLE_EXPERIMENTAL)
target_link_libraries(nv_cluster_builder_tests nv_cluster_builder gtest_main gmock_main)

if(MSVC)
    target_compile_options(nv_cluster_builder_tests PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:/W4>
      $<$<COMPILE_LANGUAGE:CXX>:/WX>
    )
    target_compile_definitions(nv_cluster_builder_tests PRIVATE WIN32_LEAN_AND_MEAN=1 NOMINMAX)
else()
    target_compile_options(nv_cluster_builder_tests PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-Wall>
      $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
      $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
      $<$<COMPILE_LANGUAGE:CXX>:-Wshadow>
      $<$<COMPILE_LANGUAGE:CXX>:-Wconversion>
      $<$<COMPILE_LANGUAGE:CXX>:-Werror>
    )
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_definitions(nv_cluster_builder_tests PRIVATE
            $<$<CONFIG:Debug>:_GLIBCXX_ASSERTIONS>
            $<$<CONFIG:Debug>:_GLIBCXX_DEBUG>           # DANGER: ABI breaking!
            $<$<CONFIG:Debug>:_GLIBCXX_DEBUG_BACKTRACE> # DANGER: ABI breaking!
        )
    endif()
endif()

include(GoogleTest)
gtest_discover_tests(nv_cluster_builder_tests)
