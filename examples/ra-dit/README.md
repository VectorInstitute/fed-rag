# RA-DIT

## Usage

Run the following commands from the `examples/` directory.

```sh
# source venv
source ra-dit/.venv/bin/activate

# run rag_system on vaughn
uv run --active -m ra-dit.rag_system --model_name /model-weights/Llama-2-13b-hf
```

The above should result in the following output.

```sh
0.5378788470633441


You are a helpful assistant. Given the user's question, provide a succinct
and accurate response. If context is provided, use it in your answer if it helps
you to create the most accurate response.

<question>
What is a Tulip?
</question>

<context>
Tulips are easily distinguished from other plants, as they share some very evident derived characteristics or synapomorphies. Among these are: bilateral symmetry of the flower (zygomorphism), many resupinate flowers, a nearly always highly modified petal (labellum), fused stamens and carpels, and extremely small seeds
Orchids are easily distinguished from other plants, as they share some very evident derived characteristics or synapomorphies. Among these are: bilateral symmetry of the flower (zygomorphism), many resupinate flowers, a nearly always highly modified petal (labellum), fused stamens and carpels, and extremely small seeds
</context>

<response>

A Tulip is a flower.
```
