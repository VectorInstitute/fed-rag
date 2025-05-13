# Generator Fine-Tuning with RALT

The overall success of RAG systems depends heavily on the generator model's ability
to effectively utilize retrieved knowledge. This capability is crucial because
LLM generators are typically trained on corpora that differ from the sources populating
the knowledge store. As a result, generators may struggle to properly integrate
or reason with retrieved information that contains domain-specific terminology,
formatting, or content structures.

Retriever-Augmented LM Training (RALT), introduced by Lin et al. (2023)[^1],
addresses this challenge by fine-tuning the generator model specifically on examples
that incorporate retrieved knowledge nodes. This process helps the generator learn
to better contextualize, interpret, and integrate information from the knowledge
store into its responses (and, even learn to ignore it when it deems it irrelevant
to the original query).

## The RALT Method

## Notes on the FedRAG Implementation of RALT

<!-- References -->
[^1]: Lin, Xi Victoria, et al. "Ra-dit: Retrieval-augmented dual instruction tuning."
  The Twelfth International Conference on Learning Representations. 2023.
