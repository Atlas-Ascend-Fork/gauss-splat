#!/bin/bash

set +e
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
ARCH=$(uname -m)
case "${ARCH}" in
    x86_64|aarch64)
        ;;
    arm64)
        ARCH=aarch64
        ;;
    *)
        echo "Unsupported architecture: ${ARCH}" >&2
        exit 1
        ;;
esac
CANN_ARCH_DIR=${ARCH}-linux

_PYTHON_SITE=$(cd /tmp && python -c "import gauss_splat, os; print(os.path.dirname(os.path.abspath(gauss_splat.__file__)))" 2>/dev/null)
if [ -n "$_PYTHON_SITE" ]; then
    BUILD_DIR=${_PYTHON_SITE}/packages/vendors/customize
else
    PYTHON_BUILD_TAG=$(python -c 'import sysconfig, sys; print(f"{sysconfig.get_platform()}-{sys.implementation.cache_tag}")' 2>/dev/null)
    if [ -z "${PYTHON_BUILD_TAG}" ]; then
        echo "Unable to determine the Python build directory." >&2
        exit 1
    fi
    BUILD_DIR=${PROJECT_ROOT}/build/lib.${PYTHON_BUILD_TAG}/gauss_splat/packages/vendors/customize
fi
CANN_PATH=$ASCEND_TOOLKIT_HOME

GS_PACKAGES=${BUILD_DIR}
OPAPI_INC=${GS_PACKAGES}/op_api/include
OPAPI_LIB=${GS_PACKAGES}/op_api/lib
OPTILING_LIB=${GS_PACKAGES}/op_impl/ai_core/tbe/op_tiling/lib/linux/${ARCH}
OPPROTO_LIB=${GS_PACKAGES}/op_proto/lib/linux/${ARCH}
CANN_PLATFORM_LIB=${CANN_PATH}/${CANN_ARCH_DIR}/lib64
CANN_OPTILING_LIB=${CANN_PATH}/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/${ARCH}
CANN_OPAPI_LIB=${CANN_PATH}/opp/built-in/op_impl/ai_core/tbe/op_api/lib/linux/${ARCH}

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
    -L${CANN_PLATFORM_LIB} \
    -L${CANN_OPTILING_LIB} \
    -L${CANN_OPAPI_LIB} \
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

