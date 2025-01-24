# Copyright (c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)
project(nv_cluster_builder)

option(NVCLUSTER_MULTITHREADED "Use parallel algorithms to generate clusters." ON)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/_install" CACHE PATH "Output location for the INSTALL target" FORCE)
endif()

file(GLOB SOURCES src/*.h src/*.hpp src/*.c src/*.cpp)
file(GLOB HEADERS include/nvcluster/*.h include/nvcluster/*.hpp)


source_group("public_include" FILES ${HEADERS})
source_group("source" FILES ${SOURCES})

add_library(nv_cluster_builder ${SOURCES} ${HEADERS})
target_compile_features(nv_cluster_builder PUBLIC cxx_std_20)
target_include_directories(nv_cluster_builder PUBLIC include)
target_include_directories(nv_cluster_builder PRIVATE src)

# All the warnings. Branch on COMPILE_LANGUAGE to avoid passing unknowns to nvcc
if(MSVC)
    target_compile_options(nv_cluster_builder PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
        #$<$<COMPILE_LANGUAGE:CXX>:/WX>
    )
    target_compile_definitions(nv_cluster_builder PRIVATE WIN32_LEAN_AND_MEAN=1 NOMINMAX)
else()
    target_compile_options(nv_cluster_builder PRIVATE
        -fno-math-errno
        -fno-trapping-math
        #-funsafe-math-optimizations
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
        $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
        $<$<COMPILE_LANGUAGE:CXX>:-Wconversion>
        $<$<COMPILE_LANGUAGE:CXX>:-Werror>
    )
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_definitions(nv_cluster_builder PRIVATE
            $<$<CONFIG:Debug>:_GLIBCXX_ASSERTIONS>
            # Do not use ABI breaking _GLIBCXX_DEBUG or _GLIBCXX_DEBUG_BACKTRACE
        )
    endif()
endif()

if(NOT NVCLUSTER_MULTITHREADED)
  target_compile_definitions(nv_cluster_builder PRIVATE -DNVCLUSTER_MULTITHREADED=0)
endif()

if(NOT MSVC)
  # Optional TBB for std::execution on linux
  find_library(TBB_LIBRARIES NAMES tbb HINTS ${TBB_DIR})
  if(TBB_LIBRARIES)
      message(STATUS "TBB: ${TBB_LIBRARIES}")
      target_link_libraries(nv_cluster_builder PRIVATE ${TBB_LIBRARIES})
  else()
      message(STATUS "TBB not found for std::execution")
  endif()
endif()

if(BUILD_TESTING)
  option(BUILD_NV_CLUSTER_BUILDER_TESTING "Build nv_cluster_builder tests" ON)
  if(BUILD_NV_CLUSTER_BUILDER_TESTING)
    enable_testing()
    add_subdirectory(test)
  endif()
endif()

install(TARGETS nv_cluster_builder)
