#! /bin/zsh
# pypipublish.sh
# Usage: From root of semantic_retrieval repo, run ./scripts/pypipublish.sh

# NOTE: This assumes you have the semantic_retrieval conda environment created.
# You will be prompted for a username and password. For the username, use __token__. 
# For the password, use the token value, including the pypi- prefix.
# To get a PyPi token, go here: 
# Need to get a token from here (scroll down to API Tokens): https://pypi.org/manage/account/ 
# If you have issues, read the docs: https://packaging.python.org/en/latest/tutorials/packaging-projects/ 

cd python 
rm -rf ./dist
conda activate semantic_retrieval
python3 -m pip install --upgrade build
python3 -m build
python3 -m pip install --upgrade twine

# If you want to upload to testpypi, uncomment the twine command below & comment out the other twine upload command.
# Note that the testpypi repo has a 'semantic-retrieval' package, so change name in pyproject.toml to 'semantic-retrieval-python-test' or something

# python3 -m twine upload --repository testpypi dist/*

python3 -m twine upload dist/*

cd ..
