<!-- markdownlint-disable-file MD024 -->

# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## Unreleased

- ...

## [0.0.4] - 2025-03-14

### Added


# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## Unreleased

- ...

## [0.0.4] - 2025-03-14

### Added

- [EXAMPLE] Adds RA-DIT retriever train loop using HuggingFace integrations (#64)
- Added HuggingFaceFLTask for federated learning with HuggingFace models (#61)
- Added federate.tester.huggingface and associated inspectors (#59)
- Added federate.trainer.huggingface decorators (#56)
- Added ability to specify device on load for HF Models (#55)

## [0.0.3] - 2025-03-12

### Added

- Build RAG system implementation (#50)
- Added RAGSystem class (#45)
- Implemented InMemoryKnowledgeStore (#43)
- Added BaseKnowledgeStore and KnowledgeNode models (#41)
- Added HFSentenceTransformerRetriever (#38)
- Added BaseRetriever class (#35)
- Added HFPreTrainedGenerator class for generative models (#34)
- Implemented BaseGenerator and HFPretrainedModelGenerator classes (#33)
- Added support for llama3 models (#30)
- Added example Retrieval-Augmented Generation with LLM integration (#28)
- Implemented DragonRetriever for example RAG system (#26)

### Changed

- Updated to use HFSentenceTransformerRetriever in examples (#48)

## [0.0.2] - 2025-03-01

### Changed

- Quickstart as a workspace member, smaller package builds (#24)

## [0.0.1] - 2025-03-01

### Added

- Working QuickStart! (#20)
- Implementation PyTorchFLTask.server() (#17)
- PyTorchFLTask and PyTorchFlowerClient (#16)
- Inspection of tester callable for pytorch.tester decorator (#14)
- federate.trainer.pytorch decorate inspects trainer loop to extract spec (#12)
- BaseTaskModel and PyTorchTaskModel (#5)
