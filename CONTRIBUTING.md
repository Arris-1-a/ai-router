# Contributing to ai-router

Thank you for considering contributing to ai-router! We welcome contributions of all kinds.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/ai-router.git`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Install dev dependencies: `pip install -e ".[dev]"`
5. Make your changes
6. Run tests: `pytest tests/ -v`
7. Run linting: `ruff check ai_router/`
8. Commit and push
9. Open a Pull Request

## Development Guidelines

### Code Style

- Follow PEP 8 with 100-character line length
- Use type hints
- Write docstrings for public APIs (Google style preferred)
- Use `ruff` for linting and formatting

### Testing

- Write tests for new features
- Maintain or improve coverage
- Run `pytest tests/ -v --cov=ai_router` before submitting

### Commit Messages

- Use clear, descriptive commit messages
- Reference issue numbers when applicable
- Example: `feat(router): add semantic routing strategy (#42)`

### Pull Requests

- Describe what your PR does and why
- Link related issues
- Ensure CI passes
- Keep PRs focused — one feature/fix per PR

## Project Structure

```
ai_router/
├── router/      # Core routing logic
├── rag/         # RAG pipeline
├── agents/      # Agent framework
├── eval/        # Evaluation tools
├── api/         # FastAPI server
└── cli.py       # CLI entry point
```

## Questions?

Open an issue or start a discussion on GitHub.
