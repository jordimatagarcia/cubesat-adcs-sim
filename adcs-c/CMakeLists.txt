cmake_minimum_required(VERSION 3.20)
project(adcs_c VERSION 0.0.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ADCS_SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(ADCS_BUILD_TESTS "Build the unit tests" ON)

set(ADCS_WARNINGS
    -Wall -Wextra -Wpedantic -Werror
    -Wconversion -Wsign-conversion -Wdouble-promotion
    -Wshadow -Wcast-qual -Wstrict-prototypes -Wmissing-prototypes
)

if(ADCS_SANITIZE)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all)
    add_link_options(-fsanitize=address,undefined)
endif()

# Sources are compiled once and linked into both a static library (benchmarks, embedded
# builds) and a shared library (loaded by the Python simulator through ctypes).
add_library(adcs_obj OBJECT src/adcs.c)
target_include_directories(adcs_obj PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(adcs_obj PRIVATE ${ADCS_WARNINGS})
set_target_properties(adcs_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(adcs_c_static STATIC $<TARGET_OBJECTS:adcs_obj>)
add_library(adcs_c_shared SHARED $<TARGET_OBJECTS:adcs_obj>)
foreach(lib adcs_c_static adcs_c_shared)
    target_include_directories(${lib} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    set_target_properties(${lib} PROPERTIES OUTPUT_NAME adcs_c)
endforeach()

if(ADCS_BUILD_TESTS)
    enable_testing()
    add_executable(test_abi tests/test_abi.c)
    target_compile_options(test_abi PRIVATE ${ADCS_WARNINGS})
    target_link_libraries(test_abi PRIVATE adcs_c_static)
    add_test(NAME abi COMMAND test_abi)
endif()
