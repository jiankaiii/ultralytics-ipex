#!/bin/bash

set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

ipex_version=2.3.1
echo "IPEX setup for version ${ipex_version}"
venv=ipex2311_xpu_venv

if [ -d "${SCRIPT_DIR}"/../"${venv}" ]; then
    echo "Venv ${venv} already exist. Please delete or rename the existing venv if you wish to start over."
    exit 0
else
    echo "Creating venv named ${venv}"
    python3 -m venv "${SCRIPT_DIR}"/../"${venv}"
fi

source "${SCRIPT_DIR}"/../"${venv}"/bin/activate
pip3 install --upgrade pip
python3 -m pip install torch==2.3.1+cxx11.abi torchvision==0.18.1+cxx11.abi torchaudio==2.3.1+cxx11.abi intel-extension-for-pytorch==2.3.110+xpu oneccl_bind_pt==2.3.100+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
pip3 install -r "${SCRIPT_DIR}"/requirements_dev.txt

echo
echo "Done"
