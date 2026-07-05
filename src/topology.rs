//! Communication `Topology` + consensus protocols.

use crate::agent::Agent;
use crate::vec2::Vec2;

/// Communication topology represented as an adjacency list.
#[derive(Debug, Clone)]
pub struct Topology {
    /// For each agent, the indices of agents it can communicate with.
    pub neighbors: Vec<Vec<usize>>,
}

impl Topology {
    /// Create a fully-connected topology for `n` agents.
    #[must_use]
    pub fn fully_connected(n: usize) -> Self {
        let neighbors = (0..n)
            .map(|i| (0..n).filter(|&j| j != i).collect())
            .collect();
        Self { neighbors }
    }

    /// Create a ring topology for `n` agents.
    #[must_use]
    pub fn ring(n: usize) -> Self {
        if n == 0 {
            return Self {
                neighbors: Vec::new(),
            };
        }
        let neighbors = (0..n)
            .map(|i| {
                if n == 1 {
                    vec![]
                } else {
                    vec![(i + n - 1) % n, (i + 1) % n]
                }
            })
            .collect();
        Self { neighbors }
    }

    /// Create a star topology with agent 0 as the hub.
    #[must_use]
    pub fn star(n: usize) -> Self {
        if n == 0 {
            return Self {
                neighbors: Vec::new(),
            };
        }
        let mut neighbors = Vec::with_capacity(n);
        neighbors.push((1..n).collect());
        for _ in 1..n {
            neighbors.push(vec![0]);
        }
        Self { neighbors }
    }

    /// Create a k-nearest-neighbor topology.
    #[must_use]
    pub fn k_nearest(agents: &[Agent], k: usize) -> Self {
        let n = agents.len();
        let neighbors = (0..n)
            .map(|i| {
                let mut dists: Vec<(usize, f64)> = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| (j, agents[i].position.distance_to(agents[j].position)))
                    .collect();
                dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                dists.iter().take(k).map(|&(j, _)| j).collect()
            })
            .collect();
        Self { neighbors }
    }

    /// Number of agents in this topology.
    #[must_use]
    pub const fn agent_count(&self) -> usize {
        self.neighbors.len()
    }

    /// Check if agent `i` can communicate with agent `j`.
    #[must_use]
    pub fn is_connected(&self, i: usize, j: usize) -> bool {
        self.neighbors.get(i).is_some_and(|ns| ns.contains(&j))
    }
}

/// Run one step of average consensus on scalar values.
pub fn consensus_step(values: &mut [f64], topology: &Topology, rate: f64) {
    let n = values.len().min(topology.agent_count());
    let deltas: Vec<f64> = (0..n)
        .map(|i| {
            let ns = &topology.neighbors[i];
            if ns.is_empty() {
                return 0.0;
            }
            #[allow(clippy::cast_precision_loss)]
            let avg = ns.iter().map(|&j| values[j]).sum::<f64>() / ns.len() as f64;
            (avg - values[i]) * rate
        })
        .collect();

    for (i, &d) in deltas.iter().enumerate() {
        values[i] += d;
    }
}

/// Run consensus on 2D vectors (e.g., position or velocity consensus).
pub fn consensus_step_vec2(values: &mut [Vec2], topology: &Topology, rate: f64) {
    let n = values.len().min(topology.agent_count());
    let deltas: Vec<Vec2> = (0..n)
        .map(|i| {
            let ns = &topology.neighbors[i];
            if ns.is_empty() {
                return Vec2::zero();
            }
            let mut sum = Vec2::zero();
            for &j in ns {
                sum += values[j];
            }
            #[allow(clippy::cast_precision_loss)]
            let avg = sum * (1.0 / ns.len() as f64);
            (avg - values[i]) * rate
        })
        .collect();

    for (i, &d) in deltas.iter().enumerate() {
        values[i] += d;
    }
}
