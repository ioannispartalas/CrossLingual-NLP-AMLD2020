echo "Checking out github repo"
INSTALL_PATH="${HOME}/projects"
echo ${INSTALL_PATH}
mkdir "${INSTALL_PATH}"
cd "${INSTALL_PATH}"
git clone https://github.com/facebookresearch/LASER.git

cd "${INSTALL_PATH}/LASER/"
export LASER="${INSTALL_PATH}/LASER"

pip install transliterate

./install_models.sh
./install_external_tools.sh
