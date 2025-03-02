from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/dragon-plus-query-encoder")
query_encoder = AutoModel.from_pretrained("facebook/dragon-plus-query-encoder")
context_encoder = AutoModel.from_pretrained(
    "facebook/dragon-plus-context-encoder"
)

# We use msmarco query and passages as an example
query = "Where was Marie Curie born?"
contexts = [
    "Maria Sklodowska, later known as Marie Curie, was born on November 7, 1867.",
    "Born in Paris on 15 May 1859, Pierre Curie was the son of Eugène Curie, a doctor of French Catholic origin from Alsace.",
]
# Apply tokenizer
query_input = tokenizer(query, return_tensors="pt")
ctx_input = tokenizer(
    contexts, padding=True, truncation=True, return_tensors="pt"
)
# Compute embeddings: take the last-layer hidden state of the [CLS] token
query_emb = query_encoder(**query_input).last_hidden_state[:, 0, :]
ctx_emb = context_encoder(**ctx_input).last_hidden_state[:, 0, :]
# Compute similarity scores using dot product
score1 = query_emb @ ctx_emb[0]  # 396.5625
score2 = query_emb @ ctx_emb[1]  # 393.8340

print(score1, score2)
