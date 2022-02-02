.PHONY: install uninstall
all: install

install:
	mkdir -p data
	mkdir -p logs
	mkdir -p models
	mkdir -p experiments
	pip install -e ./

test:
	pytest
	rm ./libs/MIA/tests/fixtures/*attack_data.h5 -f
	rm ./libs/MIA/tests/fixtures/*.json -f

uninstall:
	pip uninstall fia
