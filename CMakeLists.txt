# ── CMakeLists.txt ──────────────────────────────────────────────────
cmake_minimum_required(VERSION 3.30)
project(simdpy)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_OSX_ARCHITECTURES arm64)

find_package(pybind11 CONFIG REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# ---- build the extension -----------------------------------------------------

pybind11_add_module(simdpy simdpy_add.cpp)

target_link_libraries(
        simdpy
        PRIVATE
        pybind11::headers
        "-framework Accelerate"
)

set_target_properties(simdpy PROPERTIES PREFIX "" SUFFIX ".so")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---- test integration --------------------------------------------------------

# Enable CTest
enable_testing()

add_custom_target(pytest_run
        COMMAND ${Python3_EXECUTABLE} -m pytest -q
        ${CMAKE_CURRENT_SOURCE_DIR}/tests
        DEPENDS simdpy                 # wait for the .so to link
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_test(NAME simdpy_pytest COMMAND ${Python3_EXECUTABLE} -m pytest -q
        ${CMAKE_CURRENT_SOURCE_DIR}/tests)

add_custom_target(run_tests ALL
        DEPENDS pytest_run
)
# ────────────────────────────────────────────────────────────────────
