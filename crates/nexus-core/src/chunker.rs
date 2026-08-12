/*
 * crates/nexus-core/src/chunker.rs
 * ==================================
 * High-speed text chunking engine with sliding window token partitioning,
 * entropy calculations, and self-healing syntax protection.
 */

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Result of evaluating a streaming token boundary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryEvaluation {
    pub token_index: usize,
    pub cosine_shift: f64,
    pub token_entropy: f64,
    pub boundary_score: f64,
    pub threshold: f64,
    pub is_boundary: bool,
    pub suppressed_by_syntax: bool,
}

/// Fast text chunker splitting text by fixed size and overlap.
pub fn fast_chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    if text.is_empty() || chunk_size == 0 {
        return Vec::new();
    }

    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return Vec::new();
    }

    let mut chunks: Vec<String> = Vec::new();
    let step = if chunk_size > overlap {
        chunk_size - overlap
    } else {
        1
    };

    let mut i = 0;
    while i < words.len() {
        let end = (i + chunk_size).min(words.len());
        let chunk_words = &words[i..end];
        chunks.push(chunk_words.join(" "));
        if end == words.len() {
            break;
        }
        i += step;
    }

    chunks
}

/// Calculates directional cosine shift ΔS = 1 - cos(A, B).
pub fn compute_cosine_shift(vec_a: &[f64], vec_b: &[f64]) -> f64 {
    if vec_a.is_empty() || vec_b.is_empty() || vec_a.len() != vec_b.len() {
        return 0.0;
    }
    let dot: f64 = vec_a.iter().zip(vec_b.iter()).map(|(a, b)| a * b).sum();
    let norm_a: f64 = vec_a.iter().map(|a| a * a).sum::<f64>().sqrt();
    let norm_b: f64 = vec_b.iter().map(|b| b * b).sum::<f64>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    let cos_sim = (dot / (norm_a * norm_b)).max(-1.0).min(1.0);
    (1.0 - cos_sim).max(0.0)
}

/// Calculates conditional token Shannon entropy H(T_i | T_{i-w..i-1}) = - ∑ p log2(p).
pub fn compute_conditional_entropy(tokens: &[String]) -> f64 {
    if tokens.is_empty() {
        return 0.0;
    }
    let total = tokens.len() as f64;
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for tok in tokens {
        *counts.entry(tok.as_str()).or_insert(0) += 1;
    }

    let mut entropy = 0.0;
    for &count in counts.values() {
        let p = count as f64 / total;
        if p > 0.0 {
            entropy -= p * p.log2();
        }
    }
    entropy.max(0.0)
}

/// Tracks code block fences and nested bracket depth for self-healing syntax protection.
#[derive(Debug, Default)]
pub struct SyntaxTracker {
    pub code_fence_open: bool,
    pub curly_depth: usize,
    pub square_depth: usize,
}

impl SyntaxTracker {
    pub fn update(&mut self, token: &str) {
        if token.contains("```") {
            self.code_fence_open = !self.code_fence_open;
        }
        if !self.code_fence_open {
            for ch in token.chars() {
                match ch {
                    '{' => self.curly_depth += 1,
                    '}' => self.curly_depth = self.curly_depth.saturating_sub(1),
                    '[' => self.square_depth += 1,
                    ']' => self.square_depth = self.square_depth.saturating_sub(1),
                    _ => {}
                }
            }
        }
    }

    pub fn is_syntax_locked(&self) -> bool {
        self.code_fence_open || self.curly_depth > 0 || self.square_depth > 0
    }
}

/// High-speed Adaptive Semantic Chunker engine.
#[derive(Debug)]
pub struct AdaptiveChunkerEngine {
    pub window_size: usize,
    pub tau_boundary: f64,
    pub min_chunk_tokens: usize,
    pub max_chunk_tokens: usize,
}

impl AdaptiveChunkerEngine {
    pub fn new(window_size: usize, tau_boundary: f64, min_chunk_tokens: usize, max_chunk_tokens: usize) -> Self {
        Self {
            window_size: window_size.max(4),
            tau_boundary,
            min_chunk_tokens: min_chunk_tokens.max(1),
            max_chunk_tokens,
        }
    }

    pub fn evaluate_boundary(
        &self,
        vec_a: &[f64],
        vec_b: &[f64],
        window_b_tokens: &[String],
        syntax_locked: bool,
        token_index: usize,
    ) -> BoundaryEvaluation {
        let shift = compute_cosine_shift(vec_a, vec_b);
        let entropy = compute_conditional_entropy(window_b_tokens);
        let score = shift * entropy;
        let crosses_threshold = score > self.tau_boundary;

        let (is_boundary, suppressed_by_syntax) = if crosses_threshold {
            if syntax_locked {
                (false, true)
            } else {
                (true, false)
            }
        } else {
            (false, false)
        };

        BoundaryEvaluation {
            token_index,
            cosine_shift: shift,
            token_entropy: entropy,
            boundary_score: score,
            threshold: self.tau_boundary,
            is_boundary,
            suppressed_by_syntax,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_chunk_text() {
        let text = "one two three four five six seven eight nine ten";
        let chunks = fast_chunk_text(text, 4, 1);
        assert!(!chunks.is_empty());
    }
}
