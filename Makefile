.PHONY: test lint run install

test:
	pytest tests/

lint:
	ruff check psat/ app.py ui_components.py tests/

run:
	streamlit run app.py

install:
	pip install -e ".[dev]"
