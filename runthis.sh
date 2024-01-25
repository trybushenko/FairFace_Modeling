pip install poetry
poetry install
poetry run python -m ipykernel install --user --name=poetry-root-env
poetry run pip install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cu110/torch_nightly.html
cd data/
poetry run gdown --id 1n1wjPXLRlMrJjFdX1SxoiaO1JgwbQuFV
mv fairface-img-margin025-trainval.zip fairface_dataset.zip
poetry run pip install -e .