# Remove compiled bytecode of source files
dev-clean-pyc:
	@find . -name '*.pyc' -exec rm -f {} +
	@find . -name '*.pyo' -exec rm -f {} +
	@find . -name '__pycache__' -exec rm -fr {} +
	@find . -name '*.pytest_cache' -exec rm -fr {} +

# Setup the project environment by:
# - Run pipenv shell to start the virtual env
env:
	pipenv --python=$(conda run which python) --site-packages
	pipenv shell

# Install all libraries of package
install-all:
	pipenv install --system --dev

# Install a package 
install:
	pipenv install $(pkg)

# Install a package in dev mode
install-dev:
	pipenv install --dev $(pkg)

# Boot the game to be played by a human.
play:
	python -m bin.play

# Run the model agent.
agent:
	python -m bin.agent

# Randomly run actions in game.
sample:
	python -m bin.sample

