# git
# git clone https://nuinashco:$GITHUB_TOKEN@github.com/nuinashco/GPT2Vec.git $$ cd GPT2Vec $$ source setup_vast.sh

git config --global user.name "Ivan Havlytskyi"
git config --global user.email "ivan.havlytskyi@gmail.com"

# poetry
curl -sSL https://install.python-poetry.org | python3 -
export PATH="/root/.local/bin:$PATH"
poetry config virtualenvs.in-project true
poetry install

# login
huggingface-cli login --token $HUGGINGFACE_TOKEN
wandb login