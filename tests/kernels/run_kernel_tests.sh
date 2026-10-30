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
    PACKAGES_DIR=${_PYTHON_SITE}/packages
else
    PYTHON_BUILD_TAG=$(python -c 'import sysconfig, sys; print(f"{sysconfig.get_platform()}-{sys.implementation.cache_tag}")' 2>/dev/null)
    if [ -z "${PYTHON_BUILD_TAG}" ]; then
        echo "Unable to determine the Python build directory." >&2
        exit 1
    fi
    PACKAGES_DIR=${PROJECT_ROOT}/build/lib.${PYTHON_BUILD_TAG}/gauss_splat/packages
fi
CANN_PATH=$ASCEND_TOOLKIT_HOME

# Select the vendor installed for the target architecture: A5 builds install
# packages/vendors/customize_arch35, A2/A3 installs packages/vendors/customize.
if [ "${USE_ARCH35,,}" = "true" ] || [ "${USE_ARCH35,,}" = "on" ]; then
    VENDOR_NAME=customize_arch35
elif [ -d "${PACKAGES_DIR}/vendors/customize_arch35" ] && [ ! -d "${PACKAGES_DIR}/vendors/customize" ]; then
    VENDOR_NAME=customize_arch35
else
    VENDOR_NAME=customize
fi
BUILD_DIR=${PACKAGES_DIR}/vendors/${VENDOR_NAME}
if [ ! -d "${BUILD_DIR}" ]; then
    echo "Operator package not found: ${BUILD_DIR}" >&2
    echo "Build and install gauss-splat for the target architecture first." >&2
    exit 1
fi

GS_PACKAGES=${BUILD_DIR}
OPAPI_LIB=${GS_PACKAGES}/op_api/lib
OPTILING_LIB=${GS_PACKAGES}/op_impl/ai_core/tbe/op_tiling/lib/linux/${ARCH}
OPPROTO_LIB=${GS_PACKAGES}/op_proto/lib/linux/${ARCH}
CANN_PLATFORM_LIB=${CANN_PATH}/${CANN_ARCH_DIR}/lib64
CANN_OPTILING_LIB=${CANN_PATH}/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/${ARCH}
CANN_OPAPI_LIB=${CANN_PATH}/opp/built-in/op_impl/ai_core/tbe/op_api/lib/linux/${ARCH}

# Tests that only exist for one architecture (see build_kernel_tests.sh).
A2A3_ONLY_TESTS="test_calc_render_fwd_double_clip_gsids test_calc_render_bwd_var_clip_gsids"
A5_ONLY_TESTS="test_calc_render_fwd_clip_last_ids"
if [ "${VENDOR_NAME}" = "customize_arch35" ]; then
    SKIP_TESTS="${A2A3_ONLY_TESTS}"
else
    SKIP_TESTS="${A5_ONLY_TESTS}"
fi

export ASCEND_CUSTOM_OPP_PATH=${GS_PACKAGES}
export LD_LIBRARY_PATH=${CANN_PATH}/lib64:${CANN_PLATFORM_LIB}:${CANN_OPTILING_LIB}:${CANN_OPAPI_LIB}:${OPAPI_LIB}:${OPTILING_LIB}:${OPPROTO_LIB}:$LD_LIBRARY_PATH

WORKDIR=${SCRIPT_DIR}/build_out
mkdir -p ${WORKDIR}

RESULTS_FILE=${WORKDIR}/run_results.txt
echo "=== Kernel Test Results $(date) ===" > ${RESULTS_FILE}
echo "GS_PACKAGES: ${GS_PACKAGES}" >> ${RESULTS_FILE}
echo "" >> ${RESULTS_FILE}

for binary in ${WORKDIR}/test_*; do
    [ ! -f ${binary} ] && continue
    [[ "${binary}" == *.log ]] && continue
    [[ "${binary}" == *.txt ]] && continue
    test_name=$(basename ${binary})
    case " ${SKIP_TESTS} " in
        *" ${test_name} "*)
            echo "Skipping ${test_name} (not applicable to ${VENDOR_NAME})..."
            continue
            ;;
    esac
    echo "Running ${test_name}..."
    cd /tmp && ${binary} 2>${WORKDIR}/${test_name}_run.log
    exit_code=$?
    output=$(cat ${WORKDIR}/${test_name}_run.log 2>/dev/null)
    echo "[${test_name}] EXIT CODE: ${exit_code}"
    if [ -n "${output}" ]; then
        echo "  ${output}"
    fi
    echo "[${test_name}] EXIT CODE: ${exit_code}" >> ${RESULTS_FILE}
    echo "${output}" >> ${RESULTS_FILE}
    echo "" >> ${RESULTS_FILE}
done

echo ""
echo "====================================="
echo "=== All tests completed ==="
cat ${RESULTS_FILE}
