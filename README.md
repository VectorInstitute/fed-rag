<!-- markdownlint-disable-file MD033 MD041 -->

<img src="https://d3ddy8balm3goa.cloudfront.net/vector-fed-rag/logo-dark.svg" width="175" align="right" alt="logo"/>

# FedRAG

---------------------------------------------------------------------------------------

[![Linting](https://github.com/VectorInstitute/fed-rag/actions/workflows/lint.yml/badge.svg)](https://github.com/VectorInstitute/fed-rag/actions/workflows/lint.yml)
[![Unit Testing and Upload Coverage](https://github.com/VectorInstitute/fed-rag/actions/workflows/unit_test.yml/badge.svg)](https://github.com/VectorInstitute/fed-rag/actions/workflows/unit_test.yml)
[![codecov](https://codecov.io/github/VectorInstitute/fed-rag/graph/badge.svg?token=JjJBPckP8v)](https://codecov.io/github/VectorInstitute/fed-rag)
![GitHub License](https://img.shields.io/github/license/VectorInstitute/fed-rag)
![GitHub Release](https://img.shields.io/github/v/release/VectorInstitute/fed-rag)
[![DOI](https://zenodo.org/badge/918377874.svg)](https://doi.org/10.5281/zenodo.15092361)

FedRAG is an open-source framework for federated fine-tuning of Retrieval-Augmented
Generation (RAG) systems.

## Simplified RAG fine-tuning across federated architectures

### Federate with ease

Transform centralized RAG fine-tuning into federated learning with just a
few additional lines of code.

### Work with your tools

Seamlessly integrates with popular frameworks including HuggingFace,
LlamaIndex, and LangChain — use the tools you already know.

### Lightweight abstractions

Clean, intuitive abstractions that simplify RAG fine-tuning while
maintaining full flexibility and control.

## Installation

### From package managers

```sh
# pypi
pip install fedrag

# conda-forge
conda install -c conda-forge fed-rag
```

> [!NOTE]
> Extras for `fed-rag` are also available, such as the HuggingFace extra, which
> can be installed via `pip install fed-rag[huggingface]`

### From source

```sh
git clone https://github.com/VectorInstitute/fed-rag.git
cd fed-rag

# install using pip
pip install -e .

# or, install using uv, our package manager tool of choice
uv sync --all-extras --group dev --group docs
```

## Documentation

For more detailed documentation, visit our [official documentation site](https://vectorinstitute.github.io/fed-rag/).

- [Getting Started](https://vectorinstitute.github.io/fed-rag/getting_started/essentials/)
- [Quick Starts](https://vectorinstitute.github.io/fed-rag/getting_started/quick_starts/)
- [Examples](https://vectorinstitute.github.io/fed-rag/examples/)
- [API Reference](https://vectorinstitute.github.io/fed-rag/api_reference/)
- [Glossary](https://vectorinstitute.github.io/fed-rag/glossary/)

> [!TIP]
> This README provides a high-level overview, but our official documentation is
> updated more frequently with the latest features, tutorials, and API changes.
> For the most current information, please refer to the documentation site.

## Examples

Check out our [examples directory](examples/) for more detailed usage examples:

- Basic RAG fine-tuning with federated learning
- Implementing RA-DIT with FedRAG
- Custom federated aggregation strategies
- Integration with popular LLM frameworks

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for more details.

## Citation

If you use FedRAG in your research, please cite our library:

```bibtex
@software{Fajardo_fed-rag_2024,
author = {Fajardo, Andrei and Emerson, David},
doi = {10.5281/zenodo.15092361},
license = {Apache-2.0},
month = mar,
title = {{fed-rag}},
url = {https://github.com/VectorInstitute/fed-rag},
version = {0.0.8},
year = {2024}
}
```

> [!NOTE]
> The above citation may not reflect the most recent version of the library. We
> recommend using the Github citation widget (i.e. "Cite this respository") to
> obtain a citation entry reflecting the latest released version.

## License

FedRAG is released under the [Apache License 2.0](LICENSE).

## Acknowledgements

FedRAG is developed and maintained by the [Vector Institute](https://vectorinstitute.ai/).
