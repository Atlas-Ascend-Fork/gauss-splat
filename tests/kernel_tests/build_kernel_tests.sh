#!/bin/bash

set +e
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
BUILD_DIR=${PROJECT_ROOT}/build/packages/vendors/customize

CANN_PATH=$ASCEND_TOOLKIT_HOME

GS_PACKAGES=${BUILD_DIR}
OPAPI_INC=${GS_PACKAGES}/op_api/include
OPAPI_LIB=${GS_PACKAGES}/op_api/lib
OPTILING_LIB=${GS_PACKAGES}/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64
OPPROTO_LIB=${GS_PACKAGES}/op_proto/lib/linux/x86_64

mkdir -p ${CANN_PATH}/include/aclnnop 2>/dev/null

for f in ${OPAPI_INC}/aclnn_*.h; do
    base=$(basename $f)
    ln -sf $f ${CANN_PATH}/include/aclnnop/${base} 2>/dev/null
done

WORKDIR=${SCRIPT_DIR}/build_out
mkdir -p ${WORKDIR}

CXX=g++
CXXFLAGS="-std=c++11 -g -O0 -D_GLIBCXX_USE_CXX11_ABI=0"
INCLUDES="-I${CANN_PATH}/include -I${OPAPI_INC}"

LIB_DIRS="-L${CANN_PATH}/lib64 \
    -L${CANN_PATH}/x86_64-linux/lib64 \
    -L${CANN_PATH}/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64 \
    -L${CANN_PATH}/opp/built-in/op_impl/ai_core/tbe/op_api/lib/linux/x86_64 \
    -L${OPAPI_LIB} \
    -L${OPTILING_LIB} \
    -L${OPPROTO_LIB}"

LIBS="-Wl,--start-group -lascendcl -lnnopbase -lcust_opapi -lcust_opmaster_rt2.0 \
      -loptiling -lopapi -lregister -lexe_graph -ltiling_api -lge_common_base -lgraph \
      -lmsprofiler -Wl,--end-group -ldl -lpthread -lrt -Wl,-rpath,${CANN_PATH}/lib64 \
      -Wl,-rpath,${OPAPI_LIB} -Wl,-rpath,${OPTILING_LIB}"

echo "=== Build kernel tests ==="
echo "GS_PACKAGES: ${GS_PACKAGES}"
echo "CANN_PATH: ${CANN_PATH}"
echo ""

BUILD_LOG=${WORKDIR}/build.log
> ${BUILD_LOG}

for test_src in ${SCRIPT_DIR}/test_*.cpp; do
    test_name=$(basename ${test_src} .cpp)
    echo "Building ${test_name}..."
    ${CXX} ${CXXFLAGS} ${INCLUDES} ${test_src} -o ${WORKDIR}/${test_name} ${LIB_DIRS} ${LIBS} 2>>${BUILD_LOG}
    if [ $? -ne 0 ]; then
        echo "  [${test_name}] COMPILE FAILED"
    else
        echo "  [${test_name}] COMPILE OK"
    fi
done

echo ""
echo "=== Build completed ==="
echo "Binaries in: ${WORKDIR}/"
echo "Build log: ${BUILD_LOG}"

