cmake_minimum_required(VERSION 3.20)
project(adcs_c VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ADCS_SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(ADCS_BUILD_TESTS "Build the unit tests and the golden-vector runner" ON)
option(ADCS_USE_FLOAT "Single precision (adcs_real = float)" OFF)
option(ADCS_FUZZ "Build the libFuzzer harness (clang; implies sanitizers)" OFF)

set(ADCS_WARNINGS
    -Wall -Wextra -Wpedantic -Werror
    -Wconversion -Wsign-conversion -Wdouble-promotion
    -Wshadow -Wcast-qual -Wstrict-prototypes -Wmissing-prototypes
    -Wvla -Wstack-usage=4096
)
if(CMAKE_C_COMPILER_ID MATCHES "Clang")
    list(REMOVE_ITEM ADCS_WARNINGS -Wstack-usage=4096) # GCC only
endif()

if(ADCS_FUZZ)
    add_compile_options(-fsanitize=fuzzer-no-link,address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all)
    add_link_options(-fsanitize=address,undefined)
elseif(ADCS_SANITIZE)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all)
    add_link_options(-fsanitize=address,undefined)
endif()

set(ADCS_SOURCES
    src/adcs.c
    src/control.c
    src/flae.c
    src/linalg.c
    src/mekf.c
    src/quat.c
    src/tsukf.c
)

# Sources are compiled once and linked into both a static library (benchmarks, embedded
# builds) and a shared library (loaded by the Python simulator through ctypes). Only the
# adcs.h functions are exported from the shared library.
add_library(adcs_obj OBJECT ${ADCS_SOURCES})
target_include_directories(adcs_obj PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(adcs_obj PRIVATE ${ADCS_WARNINGS})
set_target_properties(adcs_obj PROPERTIES POSITION_INDEPENDENT_CODE ON C_VISIBILITY_PRESET hidden)
target_compile_definitions(adcs_obj PRIVATE ADCS_BUILDING_LIBRARY)
if(ADCS_USE_FLOAT)
    target_compile_definitions(adcs_obj PUBLIC ADCS_USE_FLOAT)
endif()

find_library(MATH_LIBRARY m)

add_library(adcs_c_static STATIC $<TARGET_OBJECTS:adcs_obj>)
add_library(adcs_c_shared SHARED $<TARGET_OBJECTS:adcs_obj>)
foreach(lib adcs_c_static adcs_c_shared)
    target_include_directories(${lib} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    set_target_properties(${lib} PROPERTIES OUTPUT_NAME adcs_c)
    if(MATH_LIBRARY)
        target_link_libraries(${lib} PUBLIC ${MATH_LIBRARY})
    endif()
    if(ADCS_USE_FLOAT)
        target_compile_definitions(${lib} INTERFACE ADCS_USE_FLOAT)
    endif()
endforeach()

if(ADCS_BUILD_TESTS)
    enable_testing()

    add_executable(test_unit tests/test_unit.c)
    target_compile_options(test_unit PRIVATE ${ADCS_WARNINGS})
    target_include_directories(test_unit PRIVATE src)
    target_link_libraries(test_unit PRIVATE adcs_c_static)
    add_test(NAME unit COMMAND test_unit)

    # Benchmark driver (tools/bench.sh links it against each library); built here so CI
    # keeps it compiling.
    find_package(Threads REQUIRED)
    add_executable(bench bench/bench.c tests/golden_io.c)
    target_compile_options(bench PRIVATE ${ADCS_WARNINGS})
    target_include_directories(bench PRIVATE tests)
    target_link_libraries(bench PRIVATE adcs_c_static Threads::Threads)
    add_test(NAME bench_smoke COMMAND bench --label smoke --reps 1
             ${CMAKE_CURRENT_SOURCE_DIR}/../testdata/golden/thesis/nominal.json
             ${CMAKE_CURRENT_SOURCE_DIR}/../testdata/golden/mekf/nominal.json)

    add_executable(golden_runner tests/golden_runner.c tests/golden_io.c)
    target_compile_options(golden_runner PRIVATE ${ADCS_WARNINGS})
    target_link_libraries(golden_runner PRIVATE adcs_c_static)
    # The golden tolerances are for double precision.
    if(NOT ADCS_USE_FLOAT)
        file(GLOB GOLDEN_VECTORS ${CMAKE_CURRENT_SOURCE_DIR}/../testdata/golden/*/*.json)
        foreach(vector ${GOLDEN_VECTORS})
            get_filename_component(name ${vector} NAME_WE)
            get_filename_component(dir ${vector} DIRECTORY)
            get_filename_component(variant ${dir} NAME)
            add_test(NAME golden/${variant}/${name} COMMAND golden_runner ${vector})
        endforeach()
    endif()
endif()

if(ADCS_FUZZ)
    add_executable(fuzz_step tests/fuzz_step.c)
    target_compile_options(fuzz_step PRIVATE ${ADCS_WARNINGS})
    target_link_libraries(fuzz_step PRIVATE adcs_c_static)
    target_link_options(fuzz_step PRIVATE -fsanitize=fuzzer)
endif()
