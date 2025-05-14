# Submitting a Pull Request

Pull requests (PRs) are the primary way to contribute code changes to FedRAG. We welcome contributions for bug fixes, documentation improvements, new features, and enhancements to existing functionality.

## Developing your Contribution

Follow these steps to create a well-structured pull request:

1. __Create a descriptive branch__

    ```sh
    git checkout -b feature/your-feature-name
    ```

    !!! tip
        Choose a branch name that reflects your contribution (e.g.,
        `fix/memory-leak`, `docs/improve-tutorials`, `feature/add-transformer-support`).

2. __Make your changes__ following our [Development Guidelines](#development-guidelines)

3. __Commit with clear messages__

    ```sh
    git commit -m "Add feature: description of changes"
    ```

    !!! tip
        Write commit messages that explain both what and why. Include issue numbers
        when applicable (e.g., "Fix #42: Resolve memory leak in vector store").

4. __Push to your fork__

    ```sh
    git push origin feature/your-feature-name
    ```

5. __Open a pull request__ against the `main` branch and fill in the provided
    PR message template.

### Documentation Contributions

When making documentation changes:

```sh
# Preview documentation changes locally
mkdocs serve
```

This launches a development server at `http://127.0.0.1:8000/`, allowing you to preview changes in real-time as you edit. We encourage screenshots or animated GIFs for UI-related changes.

### Code Review Process

Our review process ensures high-quality contributions:

- At least one maintainer will review your PR
- Address any feedback promptly and thoroughly
- Once approved, a maintainer will merge your PR
- Significant changes may require multiple reviewers
- Be responsive to comments and questions

## Development Guidelines

### Coding Style

We maintain high code quality standards through consistent style and automated tools:

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) conventions
- Use [Black](https://github.com/psf/black) for consistent formatting
- Apply [isort](https://pycqa.github.io/isort/) for organized imports
- Run [Ruff](https://github.com/astral-sh/ruff) for comprehensive linting

Our pre-commit hooks automatically enforce these standards when you commit changes.
You can also invoke these hooks manually via the following commands:

```sh
# Run formatter
make format

# Run linters
make lint
```

### Documentation

Clear documentation is essential:

- Document all public APIs using [Google docstring format](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- Update relevant documentation when modifying features
- Include practical examples to demonstrate functionality
- Ensure code comments explain "why" rather than "what"

### Testing

Comprehensive testing ensures reliability:

- Write tests for all new features and bug fixes
- Aim to maintain or improve overall test coverage
- Use [pytest](https://docs.pytest.org/) for writing clear, effective tests
- Verify all tests pass before submitting:

```sh
# Run the full test suite
make test
```

We appreciate your contributions to making FedRAG better!
