#!/bin/bash

set +e
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
BUILD_DIR=${PROJECT_ROOT}/build/packages/vendors/customize

CANN_PATH=$ASCEND_TOOLKIT_HOME

GS_PACKAGES=${BUILD_DIR}
OPAPI_LIB=${GS_PACKAGES}/op_api/lib
OPTILING_LIB=${GS_PACKAGES}/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64
OPPROTO_LIB=${GS_PACKAGES}/op_proto/lib/linux/x86_64

export ASCEND_CUSTOM_OPP_PATH=${GS_PACKAGES}
export LD_LIBRARY_PATH=${CANN_PATH}/lib64:${CANN_PATH}/x86_64-linux/lib64:${CANN_PATH}/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64:${CANN_PATH}/opp/build-in/op_impl/ai_core/tbe/op_api/lib/linux/x86_64:${OPAPI_LIB}:${OPTILING_LIB}:${OPPROTO_LIB}:$LD_LIBRARY_PATH

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
