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

```sh
uv sync --all-extras --groups dev --group docs
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

You can contribute to the FedRAG project in a variety of ways.

1. Submit
