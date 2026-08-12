/*
 * crates/nexus-core/src/rrf.rs
 * =============================
 * Reciprocal Rank Fusion (RRF) hybrid scoring engine combining vector similarity
 * ranks with graph connection weights into a single ranked output.
 */

use std::collections::HashMap;

/// Perform Reciprocal Rank Fusion (RRF) on vector-ranked items and graph-ranked items.
///
/// Formula:
///     RRF_score(d) = 1.0 / (k + rank_vector(d)) + 1.0 / (k + rank_graph(d))
///
/// Parameters
/// ----------
/// vector_ranks:
///     Item IDs ordered by vector similarity descending (rank 1 at index 0).
/// graph_ranks:
///     Item IDs ordered by graph connection weight descending (rank 1 at index 0).
/// k:
///     Smoothing constant (typically 60).
///
/// Returns
/// -------
/// Vec<(String, f32)>
///     Pairs of (item_id, rrf_score) sorted by rrf_score descending.
pub fn rrf_fusion(
    vector_ranks: &[String],
    graph_ranks: &[String],
    k: usize,
) -> Vec<(String, f32)> {
    let k_f32 = k as f32;
    let mut scores: HashMap<String, f32> = HashMap::new();

    // 1. Accumulate vector similarity rank scores
    for (rank_idx, item_id) in vector_ranks.iter().enumerate() {
        let rank = (rank_idx + 1) as f32;
        let score = 1.0 / (k_f32 + rank);
        *scores.entry(item_id.clone()).or_insert(0.0) += score;
    }

    // 2. Accumulate graph connection rank scores
    for (rank_idx, item_id) in graph_ranks.iter().enumerate() {
        let rank = (rank_idx + 1) as f32;
        let score = 1.0 / (k_f32 + rank);
        *scores.entry(item_id.clone()).or_insert(0.0) += score;
    }

    // 3. Sort items by combined RRF score descending
    let mut result: Vec<(String, f32)> = scores.into_iter().collect();
    result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rrf_fusion_combination() {
        let vec_ranks = vec!["doc_1".to_string(), "doc_2".to_string()];
        let graph_ranks = vec!["doc_2".to_string(), "doc_3".to_string()];
        let fused = rrf_fusion(&vec_ranks, &graph_ranks, 60);

        assert!(!fused.is_empty());
        // doc_2 is present in both rankings, so it should rank highest!
        assert_eq!(fused[0].0, "doc_2");
    }
}
