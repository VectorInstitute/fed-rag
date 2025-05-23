<!-- markdownlint-disable-file MD024 -->

# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## Unreleased

- ...

## [0.0.21] - 2025-05-23

### Changed

- Fix bug in `UnslothFastModelGenerator` when calling `to_peft` and LoRA adapters mismatch dtype with base (#360)
- Fix bug in `HFPreTrainedTokenizer.encode()` where shape of raw encoder result `tokenizer()` is of dimension 2 i.e., batched (#360)
- Improved `DataCollatorForRALT` final dtypes after padding is applied (#360)

## [0.0.20] - 2025-05-23

### Added

- Support `get_peft_model` in `UnslothFastModelGenerator` (#356)
- New integration: Unsloth 🦥 `UnslothFastModelGenerator` (#356)

### Changed

- Make knowledge node embedding optional and remove redundant embedding in Qdrant upserts (#353)

## [0.0.19] - 2025-05-20

### Added

- Add support for in-memory Qdrant instances (#350)
- Add `EvalError` (#339)
- Add streaming support for HF benchmarks (#337)
- [Feature] Add HuggingFaceBenchmarkMixin and first HuggingFaceMMLU benchmark (#334)
- [Feature] Add ExactMatchEvaluationMetric (#333)
- [Feature] Add base.evals module and BaseBenchmark BaseBenchmarker classes (#326)

### Changed

- Add `num_examples` to `BaseBenchmark` (#344)

## [0.0.18] - 2025-05-16

### Added

- Trainer managers to root import (#320)

### Changed

- Don't raise `MissingExtraError` for HF generators at import time (#320)
- Add public api for `fed_rag.trainers` (#320)
- Refactor `types` to `data_structures` (#319)

## [0.0.17] - 2025-05-16

### Changed

- Rip out public api, as its not standard (#317)

## [0.0.16] - 2025-05-16

### public API [0.0.2]

- added retrievers, generators, knowledge stores, and rest of types (#310)

## [0.0.15] - 2025-05-16

### Added

- `.api` public module (#307)
- New `core` module that houses `RAGSystem` (#306)

### Changed

- Refactor: move aux RAG system types to `fed_rag.types.rag` (#306)

## [0.0.14] - 2025-05-14

### Added

- `LlamaIndexBridge` (#285)
- Add base bridge mixin (#284)

## [0.0.13] - 2025-05-10

### Added

- [Feature] Add Qdrant knowledge store (sync) (#259)

### Changed

- `QdrantKnowledgeStore` align with qdrant sdk (#273)
- [Fix] Set timeout param to ~qdrant_client.QdrantClient and use contextmanager for client creation and teardown (#272)
- add load_nodes_kwargs and unit test `QdrantKnowledgeStore` (#266)
- fix count `QdrantKnowledgeStore` (#265)
- update federate fine-tune (#256)
- Move validation of the presence of trainers to BaseTrainerManager (#257)

## [0.0.12] - 2025-05-03

### Changed

- Manually handle the padding versus delegating to `DataCollatorForCausalLM` (#248)
- Ensure that `TrainingArguments.remove_unused_columns` is set to `False` (#248)

## [0.0.11] - 2025-05-03

### Added

- `HuggingFaceTrainerManager` missing implementations for preparing retriever/generator models for training (#245)

### Changed

- [Fix] LSRLoss should have first input in KL div be in log space (#245)
- [Fix] `DataCollatorForLSR` should subclass `SentenceTransformerDataCollator` (#245)
- [Fix] `DataCollatorForLSR` should require grads for retriever's for retriever scores, but not for lm scores (#245)

## [0.0.10] - 2025-05-02

### Added

- [Feature] Add HuggingFaceTrainerForRALT and associated DataCollatorForRALT (#241)
- [Feature] Add PyTorchTrainerMixin (#239)
- add trainers and data collators (#232)
- [Feature] Add fed_rag.base.data_collators and BaseDataCollator (#231)
- [Feature] Add target template to DataCollatorForLSR (#230)
- [Feature] Implement compute_loss for LSRSentenceTransformerTrainer (#229)
- [Feature] Add HuggingFaceLSRTrainer (#227)
- [Feature] Adds HuggingFaceTrainerMixin (#226)
- Add BaseTrainer (#225)
- [Feature] Add HFRAGTrainer (#220)
- [Feature] Add PyTorchRAGTrainer (#219)
- [Feature] Data collators for LSR (both huggingface and torch) (#187)
- Add exception for missing extra error (#208)

### Changed

- [refactor] Improvements to TrainerManagers and Trainers classes and adds BaseRetrieverTrainer & BaseGeneratorTrainer (#238)
- [Refactor] Move DataCollatorForLSR to new fed_rag.data_collators.huggingface module (#233)
- [chore] rename to trainer manager (#223)
- [chore] Move HFSentenceTransformerRetriever to huggingface module for consistency (#207)

## [0.0.9] - 2025-04-25

### Added

- [Feature] Add abstract method BaseGenerator.compute_target_sequence_proba (cherry picked) (#201)
- [Feature] Add LM Supervised Retriever Loss LSRLoss (#182)

### Changed

- Make huggingface generators more dry (#202)
- Improve/validate LSRLoss forward interface (#185)

## [0.0.8] - 2025-04-09

### Added

- `BaseTokenizer.encode` now retursn new type `EncoderResult` (#150)

## [0.0.7] - 2025-04-06

### Added

- ManagedInMemoryKnowledgeStore and ManagedMixin for KnowledgeStore [#135]
- Added `name` attribute to `BaseKnowledgeStore` [#135]
- Knowledge store exceptions [#135]

### Changed

- Persist and load uses `name` (`ks_id` only exists for managed version) [#135]
- Exception modules have their own respective base exception [#135]

## [0.0.6] - 2025-03-26

### Post

- Test Zenodo

### Added

- [Feature] Add utils.data module for building RAG Fine-tuning datasets given a RAGSystem and Sequence of examples (#98)
- [Feature] Adds BaseTokenizer and HFPretrainedTokenizer subclass (#99)
- [Feature] Enable BaseGenerator and BaseRetriever used in RAGSystem to access the rag system (#95)

### Changed

- Small fixes to problems found while running the quick start example (#93)

### Other

- Update LICENSE (#92)

## [0.0.5] - 2025-03-17

### Added

- [Feature] Added HFPeftModelGenerator for efficient fine-tuning (#84)
- [Feature] Added HuggingFace's peft.PeftModel as an acceptable model type for decoration (#76)
- Added vector compute submodule (#70)
- [Example] Added RA-DIT implementation using HFPeftModelGenerator for llama-2-7b (#86)
- [Example] Added mock generator training loop; federated for a 350M model (#67)

### Changed

- [Example] RA-DIT default generator model to load with device map auto (#88)
- [Example] RA-DIT re-factor example for better organization (#87)
- [Feature] Updated set_weights and get_weights for HuggingFaceFlowerClient to include logic for PeftModel (#82)
- Re-organized RA-DIT example app (#69)

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
