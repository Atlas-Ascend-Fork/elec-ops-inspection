# Copyright (c) 2026 FalseShare Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# Apache License Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

cmake_minimum_required(VERSION 3.16.0)

project(custom_project)

if(NOT DEFINED CANNJUDGE_COMPILE_JOBS)
    set(CANNJUDGE_COMPILE_JOBS 4)
endif()

set(ASCEND_COMPUTE_UNIT ascend910b)
find_package(ASC REQUIRED)

set(package_name custom)

npu_op_package(${package_name}
    TYPE RUN
    CONFIG
        INSTALL_PATH ${CMAKE_BINARY_DIR}
)

add_subdirectory(op_host)
add_subdirectory(op_kernel)
