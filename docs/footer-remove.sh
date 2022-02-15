PIP_PATH=$(which pip)
FILE_PATH="lib/python3.8/site-packages/furo/theme/furo/page.html"
PIP_PATH="${PIP_PATH/bin\/pip/$FILE_PATH}"
sed -i '131,138d' $PIP_PATH