/*
 * crates/nexus-core/src/vector.rs
 * ================================
 * Fast vector operations: cosine distance calculations, dot products,
 * and batch matrix similarities.
 */

/// Calculate cosine distance between a 1D query vector and a list of document vectors.
/// Cosine distance = 1.0 - cosine_similarity.
pub fn compute_cosine_distances(query_vec: &[f32], doc_vecs: &[Vec<f32>]) -> Vec<f32> {
    if query_vec.is_empty() {
        return vec![1.0; doc_vecs.len()];
    }

    let query_norm: f32 = query_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if query_norm == 0.0 {
        return vec![1.0; doc_vecs.len()];
    }

    doc_vecs
        .iter()
        .map(|doc| {
            if doc.len() != query_vec.len() {
                return 1.0;
            }
            let dot: f32 = query_vec.iter().zip(doc.iter()).map(|(q, d)| q * d).sum();
            let doc_norm: f32 = doc.iter().map(|x| x * x).sum::<f32>().sqrt();
            if doc_norm == 0.0 {
                1.0
            } else {
                let sim = (dot / (query_norm * doc_norm)).max(-1.0).min(1.0);
                (1.0 - sim).max(0.0)
            }
        })
        .collect()
}

/// Compute pairwise cosine similarity matrix for a slice of vectors.
pub fn compute_pairwise_cosine_similarity(vecs: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let n = vecs.len();
    let mut matrix = vec![vec![0.0f32; n]; n];

    let norms: Vec<f32> = vecs
        .iter()
        .map(|v| v.iter().map(|x| x * x).sum::<f32>().sqrt())
        .collect();

    for i in 0..n {
        matrix[i][i] = 1.0;
        for j in (i + 1)..n {
            if vecs[i].len() != vecs[j].len() || norms[i] == 0.0 || norms[j] == 0.0 {
                matrix[i][j] = 0.0;
                matrix[j][i] = 0.0;
                continue;
            }
            let dot: f32 = vecs[i].iter().zip(vecs[j].iter()).map(|(a, b)| a * b).sum();
            let sim = (dot / (norms[i] * norms[j])).max(-1.0).min(1.0);
            matrix[i][j] = sim;
            matrix[j][i] = sim;
        }
    }
    matrix
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_vector_zero_distance() {
        let q = vec![1.0, 0.0, 0.0];
        let docs = vec![vec![1.0, 0.0, 0.0]];
        let dists = compute_cosine_distances(&q, &docs);
        assert!((dists[0] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_orthogonal_vector_unit_distance() {
        let q = vec![1.0, 0.0];
        let docs = vec![vec![0.0, 1.0]];
        let dists = compute_cosine_distances(&q, &docs);
        assert!((dists[0] - 1.0).abs() < 1e-5);
    }
}
