# Copyright 2026 Huawei Technologies Co., Ltd
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
REPO_DIR="$(dirname "${SCRIPT_PATH}")"
BUILD_PACKAGES_DIR="${REPO_DIR}/gauss_splat/packages"
SUPPORTED_PY_VERSION=(3.12 3.13 3.14)
PY_VERSION='3.12'
SINGLE_OP=''
BUILD_TYPE='Release'
USE_ARCH35='false'

function check_python_version() {
    matched_py_version='false'
    for ver in "${SUPPORTED_PY_VERSION[@]}"; do
        if [ "${PY_VERSION}" = "${ver}" ]; then
            matched_py_version='true'
            return 0
        fi
    done
    if [ "${matched_py_version}" = 'false' ]; then echo "${PY_VERSION} is an unsupported python version, we suggest ${SUPPORTED_PY_VERSION[*]}"
        exit 1
    fi
}
function usage() {
    echo "Usage: $0 --python=3.12 [--debug] [--a5]" 1>&2
}
function parse_script_args() {
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --python=*)
                PY_VERSION="${1#*=}"
                shift 1
                ;;
            --debug)
                BUILD_TYPE='Debug'
                shift 1
                ;;
            --a5|--A5)
                USE_ARCH35='true'
                shift 1
                ;;
            *)
            usage
            return 1
            ;;
        esac
    done
    return 0
}

function main()
{
    if [ -z "${ASCEND_OPP_PATH}" ]; then
        echo "ImportError: libhccl.so: cannot open shared object file: No such file or directory. Please check that the cann package is installed. Please run 'source set_env.sh' in the CANN installation path."
        exit 1
    else
        echo "ASCEND_OPP_PATH = ${ASCEND_OPP_PATH}"
    fi

    if ! parse_script_args "$@"; then
        echo "Failed to parse script args. Please check your inputs."
        exit 1
    fi

    check_python_version

    export BUILD_PYTHON_VERSION="${PY_VERSION}"
    export USE_ARCH35

    if [ -z "${SCRIPT_PATH}" ] || [ ! -f "${SCRIPT_PATH}" ]; then
        echo "failed to resolve build.sh path: '${SCRIPT_PATH}'" 1>&2
        exit 1
    fi
    if [ -z "${REPO_DIR}" ] || [ "${REPO_DIR}" = "/" ] || [ ! -f "${REPO_DIR}/setup.py" ] || [ ! -d "${REPO_DIR}/gauss_splat" ]; then
        echo "invalid repository directory: '${REPO_DIR}'" 1>&2
        exit 1
    fi
    rm -rf -- "${BUILD_PACKAGES_DIR}"

    "python${PY_VERSION}" setup.py bdist_wheel
    if [ $? -ne 0 ]; then
        echo "Failed to compile the wheel file. Please check the source code by yourself."
        exit 1
    fi

    exit 0
}

main "$@"
