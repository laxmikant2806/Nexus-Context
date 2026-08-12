# Installation Guide

`nexus-context` can be installed as a standard Python package or compiled from source with native Rust acceleration.

---

## Prerequisites

- **Python**: `>= 3.11`
- **Rust Toolchain (Optional for source compilation)**: `cargo >= 1.75`

---

## 1. Installation via pip (PyPI)

```bash
pip install nexus-context
```

---

## 2. Installation from Source

Clone the repository and install in editable mode:

```bash
git clone https://github.com/laxmikant2806/Nexus-Context.git
cd Nexus-Context
pip install -e .
```

---

## 3. Building Rust Core Extension with Maturin

```bash
pip install maturin
maturin develop --manifest-path crates/nexus-core/Cargo.toml
```

To test if Rust acceleration is enabled in Python:

```python
from context_nexus import is_rust_available
print("Rust acceleration active:", is_rust_available())
```
