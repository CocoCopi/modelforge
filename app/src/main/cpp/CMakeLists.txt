cmake_minimum_required(VERSION 3.22.1)
project(modelforge CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(MF_CORE_SOURCES
    core/gpt.cpp
)

if(ANDROID)
    add_library(modelforge SHARED
        jni.cpp
        ${MF_CORE_SOURCES}
    )
    target_include_directories(modelforge PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_options(modelforge PRIVATE -O3 -ffast-math -fno-finite-math-only -pthread)
    target_link_libraries(modelforge PRIVATE android log)
    if(ANDROID_ABI STREQUAL "arm64-v8a")
        target_compile_definitions(modelforge PRIVATE MF_ARM64=1)
        set_target_properties(modelforge PROPERTIES COMPILE_FLAGS "-march=armv8-a")
    endif()
else()
    enable_testing()
    find_package(Threads REQUIRED)

    function(mf_add_host_test name source)
        add_executable(${name} ${source} ${MF_CORE_SOURCES})
        target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        target_compile_options(${name} PRIVATE -O2 -ffast-math -fno-finite-math-only -pthread)
        target_link_libraries(${name} PRIVATE Threads::Threads)
        add_test(NAME ${name} COMMAND ${name})
        set_tests_properties(${name} PROPERTIES TIMEOUT 300)
    endfunction()

    mf_add_host_test(test_config tests/test_config.cpp)
    mf_add_host_test(test_grad tests/test_grad.cpp)
    mf_add_host_test(test_overfit tests/test_overfit.cpp)
    mf_add_host_test(test_packed tests/test_packed.cpp)
    mf_add_host_test(test_decode tests/test_decode.cpp)
    mf_add_host_test(test_concurrent tests/test_concurrent.cpp)
    mf_add_host_test(test_export tests/test_export.cpp)

    add_executable(mf_train tools/mf_train.cpp ${MF_CORE_SOURCES})
    target_include_directories(mf_train PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_options(mf_train PRIVATE -O3 -ffast-math -fno-finite-math-only -pthread)
    target_link_libraries(mf_train PRIVATE Threads::Threads)

    add_executable(bench_matmul tests/bench_matmul.cpp ${MF_CORE_SOURCES})
    target_include_directories(bench_matmul PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_options(bench_matmul PRIVATE -O3 -ffast-math -fno-finite-math-only -pthread)
    target_link_libraries(bench_matmul PRIVATE Threads::Threads)
endif()
