/*
 * crates/nexus-core/src/lib.rs
 * =============================
 * Native Rust parsing & search acceleration engine for nexus-context.
 * Exposes PyO3 bindings for vector math, graph traversal, RRF fusion, and adaptive text chunking.
 */

pub mod chunker;
pub mod graph;
pub mod rrf;
pub mod vector;

pub use chunker::{
    compute_conditional_entropy, compute_cosine_shift, fast_chunk_text, AdaptiveChunkerEngine,
    BoundaryEvaluation, SyntaxTracker,
};
pub use graph::{traverse_graph, GraphTopology};
pub use rrf::rrf_fusion;
pub use vector::{compute_cosine_distances, compute_pairwise_cosine_similarity};

// ---------------------------------------------------------------------------
// PyO3 Bindings (Optional Python extension module feature)
// ---------------------------------------------------------------------------

#[cfg(feature = "extension-module")]
use pyo3::prelude::*;

#[cfg(feature = "extension-module")]
#[pyfunction]
fn py_compute_cosine_distances(
    query_vec: Vec<f32>,
    doc_vecs: Vec<Vec<f32>>,
) -> PyResult<Vec<f32>> {
    Ok(compute_cosine_distances(&query_vec, &doc_vecs))
}

#[cfg(feature = "extension-module")]
#[pyfunction]
fn py_traverse_graph(
    nodes: Vec<String>,
    edges: Vec<(String, String)>,
    start_node: String,
    depth: usize,
) -> PyResult<Vec<String>> {
    Ok(traverse_graph(&nodes, &edges, &start_node, depth))
}

#[cfg(feature = "extension-module")]
#[pyfunction]
fn py_rrf_fusion(
    vector_ranks: Vec<String>,
    graph_ranks: Vec<String>,
    k: usize,
) -> PyResult<Vec<(String, f32)>> {
    Ok(rrf_fusion(&vector_ranks, &graph_ranks, k))
}

#[cfg(feature = "extension-module")]
#[pyfunction]
fn py_fast_chunk_text(text: String, chunk_size: usize, overlap: usize) -> PyResult<Vec<String>> {
    Ok(fast_chunk_text(&text, chunk_size, overlap))
}

#[cfg(feature = "extension-module")]
#[pymodule]
fn nexus_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_compute_cosine_distances, m)?)?;
    m.add_function(wrap_pyfunction!(py_traverse_graph, m)?)?;
    m.add_function(wrap_pyfunction!(py_rrf_fusion, m)?)?;
    m.add_function(wrap_pyfunction!(py_fast_chunk_text, m)?)?;
    Ok(())
}
