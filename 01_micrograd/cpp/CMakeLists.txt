# 階段 1 的 C++ 版。核心實作（include/engine.hpp、engine.cpp、include/nn.hpp、nn.cpp）由學習者撰寫。
# 測試用的 golden 值由 tools/gen_golden.py 以 PyTorch float64 產生
set(MICROGRAD_GOLDEN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests/golden)

add_executable(micrograd_tests tests/test_harness.cpp)
target_link_libraries(micrograd_tests PRIVATE GTest::gtest_main nlohmann_json::nlohmann_json)
target_compile_definitions(micrograd_tests PRIVATE GOLDEN_DIR="${MICROGRAD_GOLDEN_DIR}")
target_compile_options(micrograd_tests PRIVATE -Wall -Wextra)

gtest_discover_tests(micrograd_tests)

# 實作還沒寫的時候先跳過對應的測試並明確提示，否則編譯錯誤會蓋掉 harness 測試的結果。
# 注意：這是 configure 時檢查的，新增 include/engine.hpp、engine.cpp、include/nn.hpp 或 nn.cpp 之後要重跑 cmake。
function(micrograd_file_size path out)
  if(EXISTS ${path})
    file(SIZE ${path} size)
  else()
    set(size 0)
  endif()
  set(${out} ${size} PARENT_SCOPE)
endfunction()

micrograd_file_size(${CMAKE_CURRENT_SOURCE_DIR}/include/engine.hpp MICROGRAD_HEADER_SIZE)
micrograd_file_size(${CMAKE_CURRENT_SOURCE_DIR}/include/nn.hpp MICROGRAD_NN_HEADER_SIZE)

if(MICROGRAD_HEADER_SIZE GREATER 0)
  # .cpp 存在時才編進 library；兩個都沒有時（header-only）用 INTERFACE library
  set(MICROGRAD_SOURCES)
  foreach(src engine.cpp nn.cpp)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${src})
      list(APPEND MICROGRAD_SOURCES ${src})
    endif()
  endforeach()
  if(MICROGRAD_SOURCES)
    add_library(micrograd_cpp STATIC ${MICROGRAD_SOURCES})
    target_include_directories(micrograd_cpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_options(micrograd_cpp PRIVATE -Wall -Wextra)
  else()
    add_library(micrograd_cpp INTERFACE)
    target_include_directories(micrograd_cpp INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  endif()

  add_executable(micrograd_engine_tests tests/test_engine.cpp)
  target_link_libraries(micrograd_engine_tests PRIVATE micrograd_cpp GTest::gtest_main nlohmann_json::nlohmann_json)
  target_compile_definitions(micrograd_engine_tests PRIVATE GOLDEN_DIR="${MICROGRAD_GOLDEN_DIR}")
  target_compile_options(micrograd_engine_tests PRIVATE -Wall -Wextra)
  gtest_discover_tests(micrograd_engine_tests)

  if(MICROGRAD_NN_HEADER_SIZE GREATER 0)
    add_executable(micrograd_nn_tests tests/test_nn.cpp)
    target_link_libraries(micrograd_nn_tests PRIVATE micrograd_cpp GTest::gtest_main nlohmann_json::nlohmann_json)
    target_compile_definitions(micrograd_nn_tests PRIVATE GOLDEN_DIR="${MICROGRAD_GOLDEN_DIR}")
    target_compile_options(micrograd_nn_tests PRIVATE -Wall -Wextra)
    gtest_discover_tests(micrograd_nn_tests)
  else()
    message(STATUS "01_micrograd/cpp/include/nn.hpp 還不存在或是空的，略過 micrograd_nn_tests")
  endif()
else()
  message(STATUS "01_micrograd/cpp/include/engine.hpp 還不存在或是空的，略過 micrograd_engine_tests 與 micrograd_nn_tests")
endif()
