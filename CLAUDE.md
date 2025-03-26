# CLAUDE.md: SmolModels Coding Reference

## Build/Run Commands
- Install deps: `poetry install`
- Format code: `poetry run black .`
- Lint code: `poetry run ruff check . --fix`
- Run all tests: `poetry run pytest tests/`
- Run single test: `poetry run pytest tests/path/to/test_file.py::test_function_name`
- Run unit tests: `poetry run pytest tests/unit/`
- Run integration tests: `poetry run pytest tests/integration/`
- Run with coverage: `poetry run pytest --cov=smolmodels tests/`

## Code Style
- **Formatting**: Black with 120 char line length
- **Linting**: Ruff with E203/E501/E402 ignored
- **Typing**: Use type hints and Pydantic models
- **Naming**: snake_case (functions/vars), CamelCase (classes)
- **Imports**: Group stdlib, third-party, then local imports
- **__init__.py**: No code in __init__.py files except in smolmodels/__init__.py for convenience
- **Docstrings**: Required for public modules/classes/functions; Sphinx style without type hints
- **Testing**: Write pytest tests for all new functionality

## Commit Messages
- Format: `<type>: <subject>`
- Types: feat, fix, docs, style, refactor, test, chore
- Example: `feat: add support for deepseek`
