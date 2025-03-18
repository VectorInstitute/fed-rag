# RA-DIT

## Usage

Run the following commands from the `examples/ra-dit` directory.

```sh
# source venv
source .venv/bin/activate

# run federated learning

## start server
CUDA_VISIBLE_DEVICES="-1" uv run -m ra_dit.main --task generator --generator_id llama2_7b \
--generator_variant qlora --component server

## start clients using a two-gpu setup
CUDA_VISIBLE_DEVICES=0 uv run -m ra_dit.main --task generator --generator_id \
 llama2_7b --generator_variant qlora --component client_1

CUDA_VISIBLE_DEVICES=1 uv run -m ra_dit.main --task generator --generator_id \
 llama2_7b --generator_variant qlora --component client_2
```
