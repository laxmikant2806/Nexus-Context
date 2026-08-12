/*
 * crates/nexus-core/src/graph.rs
 * ===============================
 * In-memory graph adjacency topology, BFS multi-hop traversal, and reachability.
 */

use std::collections::{HashMap, HashSet, VecDeque};

/// Fast in-memory graph structure storing nodes and directed adjacency edges.
#[derive(Debug, Default, Clone)]
pub struct GraphTopology {
    pub adjacency: HashMap<String, Vec<String>>,
}

impl GraphTopology {
    pub fn new() -> Self {
        Self {
            adjacency: HashMap::new(),
        }
    }

    pub fn add_edge(&mut self, source: String, target: String) {
        self.adjacency.entry(source).or_default().push(target);
    }

    /// Build graph from node and edge lists.
    pub fn from_tuples(_nodes: &[String], edges: &[(String, String)]) -> Self {
        let mut graph = Self::new();
        for (src, tgt) in edges {
            graph.add_edge(src.clone(), tgt.clone());
        }
        graph
    }

    /// Traverse graph using BFS up to `max_depth` hops starting from `start_node`.
    pub fn traverse(&self, start_node: &str, max_depth: usize) -> Vec<String> {
        let mut visited: HashSet<String> = HashSet::new();
        let mut result: Vec<String> = Vec::new();
        let mut queue: VecDeque<(String, usize)> = VecDeque::new();

        visited.insert(start_node.to_string());
        result.push(start_node.to_string());
        queue.push_back((start_node.to_string(), 0));

        while let Some((current, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }

            if let Some(neighbors) = self.adjacency.get(&current) {
                for neighbor in neighbors {
                    if !visited.contains(neighbor) {
                        visited.insert(neighbor.clone());
                        result.push(neighbor.clone());
                        queue.push_back((neighbor.clone(), depth + 1));
                    }
                }
            }
        }
        result
    }
}

/// Standalone function for BFS graph traversal exposed to PyO3 bindings.
pub fn traverse_graph(
    nodes: &[String],
    edges: &[(String, String)],
    start_node: &str,
    depth: usize,
) -> Vec<String> {
    let graph = GraphTopology::from_tuples(nodes, edges);
    graph.traverse(start_node, depth)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bfs_traversal() {
        let nodes = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let edges = vec![
            ("A".to_string(), "B".to_string()),
            ("B".to_string(), "C".to_string()),
        ];
        let traversed = traverse_graph(&nodes, &edges, "A", 2);
        assert_eq!(traversed, vec!["A", "B", "C"]);
    }
}
