# Ascend C 算子库 CMakeLists.txt
# 重新组织后的目录结构：每个算子一个目录，每个算子下有op_host和op_kernel子目录

# 定义算子列表
set(ASCEND_OPS
    calc_render_bwd_var_clip_gsids
    calc_render_fwd_double_clip_gsids
    calc_render_simt_bwd
    calc_render_simt_fwd
    flash_gaussian_build_mask
    fully_fused_projection_bwd
    fused_ssim_backward
    fused_ssim_forward
    gaussian_filter
    gaussian_sort
    projection_three_dims_gaussian_forward
    quat_scales_to_covars
    spherical_harmonics_bwd
    spherical_harmonics_forward
)

set(ASCEND_HOST_SRC "")
set(ASCEND_KERNEL_SRC "")

set(BUILD_ARCH35 ${USE_ARCH35})
if("ascend950" IN_LIST ASCEND_COMPUTE_UNIT)
    set(BUILD_ARCH35 ON)
endif()

if(BUILD_ARCH35)
    set(ASCEND_AUTOGEN_PATH ${CMAKE_BINARY_DIR}/autogen_arch35)
    set(vendor_name customize_arch35)
endif()
file(MAKE_DIRECTORY ${ASCEND_AUTOGEN_PATH})

set(ASCEND_OPS_TARGET "")
foreach(op_name ${ASCEND_OPS})
    # A5只使用arch35实现，A2/A3只使用arch22实现；缺失对应架构目录的算子不参与编译。
    if(BUILD_ARCH35)
        file(GLOB op_host_src
            ${CMAKE_CURRENT_SOURCE_DIR}/${op_name}/op_host/arch35/*.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/${op_name}/op_host/arch35/*.h
        )
    else()
        file(GLOB op_host_src
            ${CMAKE_CURRENT_SOURCE_DIR}/${op_name}/op_host/arch22/*.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/${op_name}/op_host/arch22/*.h
        )
    endif()
    list(APPEND ASCEND_HOST_SRC ${op_host_src})

    if(BUILD_ARCH35)
        file(GLOB op_kernel_src
            ${CMAKE_CURRENT_SOURCE_DIR}/${op_name}/op_kernel/arch35/*.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/${op_name}/op_kernel/arch35/*.h
        )
    else()
        file(GLOB op_kernel_src
            ${CMAKE_CURRENT_SOURCE_DIR}/${op_name}/op_kernel/arch22/*.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/${op_name}/op_kernel/arch22/*.h
        )
    endif()
    list(APPEND ASCEND_KERNEL_SRC ${op_kernel_src})

    if(op_host_src AND op_kernel_src)
        list(APPEND ASCEND_OPS_TARGET ${op_name})
    endif()
endforeach()

message(STATUS "Ascend host source files:")
foreach(src ${ASCEND_HOST_SRC})
    message(STATUS "  ${src}")
endforeach()

message(STATUS "Ascend kernel source files:")
foreach(src ${ASCEND_KERNEL_SRC})
    message(STATUS "  ${src}")
endforeach()

message(STATUS "Ascend operators built for this target:")
foreach(op_name ${ASCEND_OPS_TARGET})
    message(STATUS "  ${op_name}")
endforeach()

if(BUILD_STAGE EQUAL 0)
  # ===================Build Stage 0: 编译算子库==================
  add_library(ascend_all_ops SHARED ${ASCEND_HOST_SRC})
  target_compile_options(ascend_all_ops PRIVATE -g -fPIC -std=c++11
                                                -D_GLIBCXX_USE_CXX11_ABI=0)
  target_include_directories(ascend_all_ops PRIVATE ${CANN_INCLUDE_PATH})
  target_link_libraries(ascend_all_ops PRIVATE intf_pub exe_graph register
                                                 tiling_api ascendcl)
  add_custom_command(
    TARGET ascend_all_ops
    POST_BUILD
    COMMAND ${ASCEND_CANN_PACKAGE_PATH}/toolkit/tools/opbuild/op_build
            $<TARGET_FILE:ascend_all_ops> ${ASCEND_AUTOGEN_PATH})

elseif(BUILD_STAGE EQUAL 1)
  # ===================Build Stage 1: 编译 proto==================
  add_library(cust_op_proto SHARED ${ASCEND_AUTOGEN_PATH}/op_proto.cc)
  target_compile_definitions(cust_op_proto PRIVATE OP_PROTO_LIB)
  target_compile_options(cust_op_proto PRIVATE -fvisibility=hidden)
  target_link_libraries(
    cust_op_proto
    PRIVATE intf_pub
            exe_graph
            register
            tiling_api
            ascendcl
            -Wl,--whole-archive
            rt2_registry
            -Wl,--no-whole-archive)
  set_target_properties(cust_op_proto PROPERTIES OUTPUT_NAME
                                                 cust_opsproto_rt2.0)
  install_target(
    TRG cust_op_proto DST
    packages/vendors/${vendor_name}/op_proto/lib/linux/${CMAKE_SYSTEM_PROCESSOR}
  )
  install_file(TRG cust_op_proto SRC ${ASCEND_AUTOGEN_PATH}/op_proto.h DST
               packages/vendors/${vendor_name}/op_proto/inc)

  # 编译 tiling 库
  add_library(cust_optiling SHARED ${ASCEND_HOST_SRC})
  target_compile_definitions(cust_optiling PRIVATE OP_TILING_LIB)
  target_compile_options(cust_optiling PRIVATE -fvisibility=hidden)
  target_link_libraries(
    cust_optiling
    PRIVATE intf_pub
            exe_graph
            register
            tiling_api
            ascendcl
            -Wl,--whole-archive
            rt2_registry
            -Wl,--no-whole-archive)
  set_target_properties(cust_optiling PROPERTIES OUTPUT_NAME
                                                 cust_opmaster_rt2.0)
  install_target(
    TRG
    cust_optiling
    DST
    packages/vendors/${vendor_name}/op_impl/ai_core/tbe/op_tiling/lib/linux/${CMAKE_SYSTEM_PROCESSOR}
  )
  # create liboptiling.so link
  add_custom_command(
    TARGET cust_optiling
    POST_BUILD
    COMMAND
      ${CMAKE_COMMAND} -E chdir
      ${gauss_splat_PATH}/packages/vendors/${vendor_name}/op_impl/ai_core/tbe/op_tiling
      ${CMAKE_COMMAND} -E create_symlink
      lib/linux/${CMAKE_SYSTEM_PROCESSOR}/$<TARGET_FILE_NAME:cust_optiling>
      liboptiling.so)
  install(
    FILES
      ${gauss_splat_PATH}/packages/vendors/${vendor_name}/op_impl/ai_core/tbe/op_tiling/liboptiling.so
    DESTINATION packages/vendors/${vendor_name}/op_impl/ai_core/tbe/op_tiling)

  # ===================Build ACLNN API===================
  file(GLOB ACLNN_SRC ${ASCEND_AUTOGEN_PATH}/aclnn_*.cpp)
  file(GLOB ACLNN_INC ${ASCEND_AUTOGEN_PATH}/aclnn_*.h)
  add_library(cust_opapi SHARED ${ACLNN_SRC})
  if(BUILD_ARCH35)
    set_target_properties(cust_opapi PROPERTIES OUTPUT_NAME cust_opapi_arch35)
  endif()
  target_link_libraries(cust_opapi PRIVATE intf_pub ascendcl nnopbase opapi)

  install_target(TRG cust_opapi DST packages/vendors/${vendor_name}/op_api/lib)
  install_file(TRG cust_opapi SRC ${ACLNN_INC} DST
               packages/vendors/${vendor_name}/op_api/include)
  if("${CMAKE_BUILD_TYPE}x" STREQUAL "Debugx")
    add_ops_compile_options(ALL OPTIONS -g -O0)
  endif()

  # Binary compilation expects all selected kernel sources in one directory.
  # Keep A5 inputs separate so arch22 implementations cannot be mixed with arch35.
  if(BUILD_ARCH35)
    set(KERNEL_IMPL_DIR ${CMAKE_CURRENT_BINARY_DIR}/kernel_impl_arch35)
  else()
    set(KERNEL_IMPL_DIR ${CMAKE_CURRENT_BINARY_DIR}/kernel_impl_arch22)
  endif()
  file(MAKE_DIRECTORY ${KERNEL_IMPL_DIR})
  foreach(kernel_file ${ASCEND_KERNEL_SRC})
    get_filename_component(kernel_file_name ${kernel_file} NAME)
    configure_file(${kernel_file} ${KERNEL_IMPL_DIR}/${kernel_file_name} COPYONLY)
  endforeach()

  # 为每个算子生成 ops-info 和 ascendc 实现
  foreach(compute_unit ${ASCEND_COMPUTE_UNIT})
    if(EXISTS ${ASCEND_AUTOGEN_PATH}/aic-${compute_unit}-ops-info.ini)
      # 仅为目标架构编译的算子生成 ops-info
      foreach(op_name ${ASCEND_OPS_TARGET})
        add_ops_info_target(
          TARGET
          ops_info_gen_${compute_unit}_${op_name}
          OUTPUT
          ${gauss_splat_PATH}/packages/vendors/${vendor_name}/op_impl/ai_core/tbe/config/${compute_unit}/aic-${compute_unit}-ops-info-${op_name}.json
          OPS_INFO
          ${ASCEND_AUTOGEN_PATH}/aic-${compute_unit}-ops-info.ini
          INSTALL_DIR
          packages/vendors/${vendor_name}/op_impl/ai_core/tbe/config/${compute_unit})
      endforeach()

      # 生成 ascendc impl py（仅一次）
      if(NOT TARGET ascendc_impl_gen)
        foreach(op_name ${ASCEND_OPS_TARGET})
          if(BUILD_ARCH35)
            set(op_kernel_dir ${CMAKE_CURRENT_SOURCE_DIR}/${op_name}/op_kernel/arch35)
          else()
            set(op_kernel_dir ${CMAKE_CURRENT_SOURCE_DIR}/${op_name}/op_kernel/arch22)
          endif()
          add_ops_impl_target(
            TARGET
            ascendc_impl_gen_${op_name}
            OPS_INFO
            ${ASCEND_AUTOGEN_PATH}/aic-${compute_unit}-ops-info.ini
            IMPL_DIR
            ${op_kernel_dir}
            OUT_DIR
            ${gauss_splat_PATH}/packages/vendors/${vendor_name}/op_impl/ai_core/tbe/${vendor_name}_impl
          )
          set(ALL_ASCENDC_IMPL_TARGETS ${ALL_ASCENDC_IMPL_TARGETS} ascendc_impl_gen_${op_name})
        endforeach()

        add_custom_target(ascendc_impl_gen ALL DEPENDS ${ALL_ASCENDC_IMPL_TARGETS})

        install_file(
          TRG
          ascendc_impl_gen
          SRC
          ${ASCEND_KERNEL_SRC}
          DST
          packages/vendors/${vendor_name}/op_impl/ai_core/tbe/${vendor_name}_impl/dynamic
        )
      endif()

      # dynamic shape binary compile
      if(${ENABLE_BINARY_PACKAGE})
        add_bin_compile_target(
          TARGET
          ascendc_bin_${compute_unit}
          OPS_INFO
          ${ASCEND_AUTOGEN_PATH}/aic-${compute_unit}-ops-info.ini
          IMPL_DIR
          ${KERNEL_IMPL_DIR}
          ADP_DIR
          ${gauss_splat_PATH}/packages/vendors/${vendor_name}/op_impl/ai_core/tbe/${vendor_name}_impl/dynamic
          OUT_DIR
          ${CMAKE_CURRENT_BINARY_DIR}/binary/${compute_unit}
          KERNEL_DIR
          ${gauss_splat_PATH}/packages/vendors/${vendor_name}/op_impl/ai_core/tbe/kernel
          INSTALL_DIR
          packages/vendors/${vendor_name}/op_impl/ai_core/tbe/kernel
          COMPUTE_UNIT
          ${compute_unit})
        add_dependencies(ascendc_bin_${compute_unit} ascendc_impl_gen
                         cust_optiling)
      endif()
    endif()
  endforeach()

  # generate npu_supported_ops.json
  add_npu_support_target(
    TARGET
    npu_supported_ops
    OPS_INFO_DIR
    ${ASCEND_AUTOGEN_PATH}
    OUT_DIR
    ${gauss_splat_PATH}/packages/vendors/${vendor_name}/op_impl/ai_core/tbe/op_info_cfg/ai_core
    INSTALL_DIR
    packages/vendors/${vendor_name}/framework/${ASCEND_FRAMEWORK_TYPE})

  get_system_info(SYSTEM_INFO)

  # gen version.info
  add_custom_target(
    gen_version_info ALL
    COMMAND
      bash ${PROJECT_SOURCE_DIR}/cmake/util/gen_version_info.sh
      ${ASCEND_CANN_PACKAGE_PATH}
      ${gauss_splat_PATH}/packages/vendors/${vendor_name})

  install(FILES ${gauss_splat_PATH}/packages/vendors/${vendor_name}/version.info
          DESTINATION packages/vendors/${vendor_name})
endif()
