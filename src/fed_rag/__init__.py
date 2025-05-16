from importlib.metadata import version

try:
    __version__ = version("fed-rag")
except ImportError:
    __version__ = "unknown"  # fallback for development installs
