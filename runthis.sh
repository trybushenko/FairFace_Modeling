pip install poetry
poetry install
poetry shell
poetry build
poetry run python -m ipykernel install --user --name=poetry-root-env
pip install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cu110/torch_nightly.html
cd data/
gdown --id 1n1wjPXLRlMrJjFdX1SxoiaO1JgwbQuFV
mv fairface-img-margin025-trainval.zip fairface_dataset.zip
