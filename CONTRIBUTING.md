<!-- markdownlint-disable-file MD041 MD029 -->

# 🌟 Contributing to FedRAG

Thank you for your interest in contributing to FedRAG! This document provides
guidelines and instructions for contributing.

We welcome contributions from developers of all skill levels. Whether you're
fixing a typo or implementing a complex feature, your help is valuable to the
FedRAG project.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](./CODE_OF_CONDUCT.md).
Please read it before contributing.

## Getting Started

In order to get going, we need to make sure you have the right development
environment setup. Below, we provide the 4 steps to get you on you way!

1. Fork and clone the repository

```sh
# Clone your fork
git clone https://github.com/YOUR_USERNAME/fed-rag.git
cd fed-rag
```

2. Setup the project's virtual environment with `uv`

You must have the `uv` Python package manager installed on your machine. For installation
instructions, you can visit uv's [official documentation](https://docs.astral.sh/uv/getting-started/installation/)
or its Github [repo](https://github.com/astral-sh/uv?tab=readme-ov-file#installation).

```sh
uv sync --all-extras --group dev --group docs
```

3. Activate the project's virtual environment

The previous command will automatically create a virtual environment stored in
`.venv` folder. To activate it, use the below command:

```sh
source .venv/bin/activate
```

4. Install pre-commit hooks

```sh
pre-commit install
```

## What Contributions Can You Make?

Everyone can contribute to FedRAG! Whether you're a seasoned ML engineer, new to
federated learning and RAG/LLMs, or somewhere in between—your input is valuable.
Here are several ways to get involved:

### 1. 🐛 Fix Bugs & Issues

Fixing bugs is a great way to get started. Check our
[GitHub Issues](https://github.com/VectorInstitute/fed-rag/issues) page and look
for issues labeled `good first issue` to begin with.

### 2. 📝 Improve Documentation

Help make our documentation more comprehensive and accessible. This includes:

- API documentation
- Usage guides
- Glossary
- Tutorials
- Architecture explanations

### 3. ✨ Add New Features

Extend FedRAG's capabilities by contributing new features that enhance both centralized
and federated fine-tuning of RAG systems. Share your ideas by opening an issue
first to discuss implementation details.

### 4. 📊 Share Examples

If you've used FedRAG in interesting ways, consider contributing:

- Example notebooks
- Case studies
- Benchmarks
- Integration examples

## Making Contributions

Once you've decided on what contribution you'd like to make, you can follow the
listed steps below to create a development branch and submit your pull request.

> [!NOTE]
> Make sure to have followed the instructions listed in the [Getting Started](#getting-started)
> section in order to get your development environment set up properly.

### Developing your Contribution and Creating a Pull Request

1. __Create a new branch__

```sh
git checkout -b feature/your-feature-name
```

2. __Make your changes__ following our [Development Guidelines](#development-guidelines).

3. __Commit your changes__ with a clear message

```sh
git commit -m "Add feature: description of changes"
```

4. __Push to your fork__

```sh
git push origin feature/your-feature-name
```

5. __Open a pull request__ against the `main` branch

### When Contributing to Docs

If you are making a contribution to the docs, then in addition to the previously
listed steps, during development you may find it helpful to watch the docs as you
make changes locally.

```sh
# in the root of the project
mkdocs serve
```

This will launch a development server on `http://127.0.0.1:8000/`. You can enter
this address in your web browser of choice to watch your changes live as they happen.

### Code Review Process

Below, we loosely describe the process of having your code reviewed and eventually
merged into `main`.

- At least one maintainer will review your PR
- Address any requested changes or feedback
- Once approved, a maintainer will merge your PR
- For significant changes, multiple reviewers may be required

## Development Guidelines

### Coding Style

We follow established Python conventions and use tools to maintain code quality:

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code style
- Use [Black](https://github.com/psf/black) for automatic code formatting
- Apply [isort](https://pycqa.github.io/isort/) for organizing imports
- Run [Ruff](https://github.com/astral-sh/ruff) for linting

The pre-commit hooks you installed will automatically check and fix many style issues.

### Documentation

Good documentation is crucial for our project:

- Document all public functions, classes, and methods using [Google docstring format](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- Keep documentation updated when changing code
- Include examples where appropriate to show usage

### Testing

We prioritize testing to maintain code quality:

- Write tests for all new features and bug fixes
- Maintain or improve test coverage
- We use [pytest](https://docs.pytest.org/) for our test framework
- Run all tests before submitting your PR:

```sh
# while in root of project and with project's virtual env active
make test
```

## Communication

We have several channels for project communication:

- For bug reports and feature requests, use [GitHub Issues](https://github.com/VectorInstitute/fed-rag/issues)
- For longer discussions, use [GitHub Discussions](https://github.com/VectorInstitute/fed-rag/discussions)

## License

By contributing to FedRAG, you agree that your contributions will be licensed
under the project's [LICENSE](./LICENSE) file.

## ✨ Final Words

We're excited to see what you'll bring to the FedRAG community! Remember that
contributions of all sizes matter - from fixing a typo to implementing complex
federated learning algorithms. Your involvement helps push the boundaries of what's
possible with federated fine-tuning of RAG.

We look forward to your pull requests!
