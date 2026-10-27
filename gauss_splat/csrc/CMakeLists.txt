file(GLOB ASCEND_CSRC_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# A5只编译SIMT渲染实现(CalcRenderSimt.cpp)，非A5只编译clip_gsids渲染实现(CalcRender.cpp)。
if(USE_ARCH35 OR "ascend950" IN_LIST ASCEND_COMPUTE_UNIT)
  set(GSPLAT_ARCH_A5 ON)
  list(REMOVE_ITEM ASCEND_CSRC_SRC ${CMAKE_CURRENT_SOURCE_DIR}/CalcRender.cpp)
else()
  set(GSPLAT_ARCH_A5 OFF)
  list(REMOVE_ITEM ASCEND_CSRC_SRC ${CMAKE_CURRENT_SOURCE_DIR}/CalcRenderSimt.cpp)
endif()

if(BUILD_STAGE EQUAL 2)
  set(Python3_USE_STATIC_LIBS FALSE)
  find_package(Python3 COMPONENTS Interpreter Development)

  # torch_npu may print driver log-init messages to stdout on import; keep
  # only output lines that are existing directories so the paths stay clean.
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
            "import os; import torch; print(os.path.dirname(torch.__file__))"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TORCH_PATH_RAW)
  string(REGEX REPLACE "\n" ";" TORCH_PATH_LINES "${TORCH_PATH_RAW}")
  foreach(_line ${TORCH_PATH_LINES})
    if(EXISTS "${_line}")
      set(TORCH_PATH "${_line}")
      break()
    endif()
  endforeach()
  execute_process(
    COMMAND
      ${Python3_EXECUTABLE} -c
      "import os; import site; print(site.getsitepackages()[0] + '/torch_npu')"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TORCH_NPU_PATH_RAW)
  string(REGEX REPLACE "\n" ";" TORCH_NPU_PATH_LINES "${TORCH_NPU_PATH_RAW}")
  foreach(_line ${TORCH_NPU_PATH_LINES})
    if(EXISTS "${_line}")
      set(TORCH_NPU_PATH "${_line}")
      break()
    endif()
  endforeach()
  message("TORCH_PATH is ${TORCH_PATH}")
  message("TORCH_NPU_PATH is ${TORCH_NPU_PATH}")

  set(EXT_CXX_FLAGS "${EXT_CXX_FLAGS}")
  separate_arguments(EXT_CXX_FLAGS)
  add_library(gsplat_c SHARED ${ASCEND_CSRC_SRC})
  if(GSPLAT_ARCH_A5)
    target_compile_definitions(gsplat_c PRIVATE ASCEND_COMPUTE_UNI_A5)
  endif()
  set_target_properties(
    gsplat_c
    PROPERTIES OUTPUT_NAME "${gauss_splat_PATH}/gsplat_c.${Python3_SOABI}"
               PREFIX ""
               SUFFIX ".so")

  if(${COMPILE_WITH_XLA})
    target_compile_definitions(gsplat_c PRIVATE COMPILE_WITH_XLA)
  endif()
  target_include_directories(
    gsplat_c
    PRIVATE ${Python3_INCLUDE_DIRS}
            ${PROJECT_SOURCE_DIR}/include
            ${TORCH_NPU_PATH}/include
            ${TORCH_NPU_PATH}/include/third_party/hccl/inc
            ${TORCH_NPU_PATH}/include/third_party/acl/inc
            ${TORCH_PATH}/include
            ${TORCH_PATH}/include/torch/csrc/api/include)
  target_compile_options(
    gsplat_c
    PRIVATE -fPIC
            $<$<CONFIG:Release>:-O3>
            $<$<CONFIG:Debug>:-O0
            -g>
            -fstack-protector-all
            -DTORCH_API_INCLUDE_EXTENSION_H
            -DTORCH_EXTENSION_NAME=gsplat_c
            -D_GLIBCXX_USE_CXX11_ABI=${ABI}
            ${EXT_CXX_FLAGS})

  target_link_directories(gsplat_c PRIVATE ${TORCH_PATH}/lib ${TORCH_NPU_PATH}/lib)
  target_link_libraries(gsplat_c PRIVATE c10 torch torch_python torch_npu)
  target_link_options(
    gsplat_c
    PRIVATE
    $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:-pie>
    $<$<CONFIG:Release>:-s>
    -Wl,-z,relro
    -Wl,-z,now
    -Wl,-z,noexecstack)
endif()