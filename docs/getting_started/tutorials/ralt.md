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

Like the [LSR method](./lsr.md), the RALT method is also applied over a training
dataset (query, response) pairs. For each training example, we first retrieve the
top-$k$ knowledge nodes, and then create $k$ independent instruction fine-tuning
examples. The instruction fine-tuning template involves placeholders for the
query, response, and the knowledge nodes' content (i.e., context for the query).

```py title="an example instruction template"
instruction_template = """You are a helpful assistant. Given the user query below,
provide a response making use of the provided background context.

<query>
{query}
</query>

<context>
{context}
</context>

<response>
{response}
</response>
"""
```

### RALT training objective

For RALT, we apply the usual masked causal language modelling task, which trains
the model to predict the next token given the previously seen tokens. Mathematically,
if we let $\{(q_i, r_i)\}_{i=1}^N$ represent the training dataset of (query, response),
pairs, and further, let $c_{i,j}$ represent the context from the $j$-th knowledge
node retrieved by the RAG system for query, $q_i$, then we can write the RALT loss
as follows:

$$
\mathcal{L}_{RALT} = - \sum_{i}^N\sum_{j}^k \log p_{LLM}(r_i | q_i \circ c_{i,j}),
$$

where $\log p_{LLM}(r_i | q_i \circ c_{i,j})$ is the log probability that response,
$r_i$, is produced by the LLM given the input sequence $q_i \circ c_{i,j}$ (with "$\circ$"
representing concatenation).

## Notes on the FedRAG Implementation of RALT

The RALT implementation in FedRAG involves the typical coordination between a data
collator and a trainer object. The `DataCollatorForRALT` takes on the responsibility
of retrieving the $k$ nodes for every query in the batch, and creating the $k$
instruction-tuning instances. Tokenization and padding are also applied in the
data collator. The `TrainerForRALT` then performs the causal language modelling
training on the collated data for the generator model.

!!! note
    An alternative implementation would be to pass the creation of the instruction
    instances from the data collator to a pre-processing step that creates a
    training dataset. In other words, the train dataset is not the (query, response)
    pair, but an already processed instruction fine-tuning dataset. For unification
    purposes, the former was chosen to promote consistency between retriever and
    generator trainer workflows.

<!-- References -->
[^1]: Lin, Xi Victoria, et al. "Ra-dit: Retrieval-augmented dual instruction tuning."
  The Twelfth International Conference on Learning Representations. 2023.
