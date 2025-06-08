# Example Scripts

This sub-directory contains example scripts that are run either using the command
line or within a Jupyter notebook (i.e., one of our cookbooks).

To generally be able to run any of these scripts you should install from source.

```sh
# clone repo
git clone git@github.com:VectorInstitute/fed-rag.git

# install from source with uv
cd fed-rag
uv sync --all-extras --dev --group docs

# run script
uv run example_scripts/<script-name>.py
```
