//! Task allocation: greedy + auction.

use std::collections::HashMap;

use crate::agent::Agent;
use crate::vec2::Vec2;

/// A task with a position and a priority.
#[derive(Debug, Clone)]
pub struct Task {
    pub id: usize,
    pub position: Vec2,
    pub priority: f64,
}

impl Task {
    #[must_use]
    pub const fn new(id: usize, position: Vec2, priority: f64) -> Self {
        Self {
            id,
            position,
            priority,
        }
    }
}

/// Result of task allocation: maps agent id -> task id.
pub type Allocation = HashMap<usize, usize>;

/// Greedy nearest-first task allocation.
/// Each task is assigned to the closest available agent, weighted by priority.
#[must_use]
pub fn allocate_greedy(agents: &[Agent], tasks: &[Task]) -> Allocation {
    let mut allocation = Allocation::new();
    let mut assigned_agents: Vec<bool> = vec![false; agents.len()];

    let mut task_order: Vec<usize> = (0..tasks.len()).collect();
    task_order.sort_by(|&a, &b| {
        tasks[b]
            .priority
            .partial_cmp(&tasks[a].priority)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for &ti in &task_order {
        let task = &tasks[ti];
        let mut best_agent = None;
        let mut best_dist = f64::MAX;

        for (ai, agent) in agents.iter().enumerate() {
            if assigned_agents[ai] {
                continue;
            }
            let d = agent.position.distance_to(task.position);
            if d < best_dist {
                best_dist = d;
                best_agent = Some(ai);
            }
        }

        if let Some(ai) = best_agent {
            allocation.insert(agents[ai].id, tasks[ti].id);
            assigned_agents[ai] = true;
        }
    }

    allocation
}

/// Auction-based task allocation.
#[must_use]
pub fn allocate_auction(agents: &[Agent], tasks: &[Task]) -> Allocation {
    let mut allocation = Allocation::new();
    let mut assigned_agents: Vec<bool> = vec![false; agents.len()];
    let mut assigned_tasks: Vec<bool> = vec![false; tasks.len()];

    let n_rounds = agents.len().min(tasks.len());

    for _ in 0..n_rounds {
        let mut best_bid = f64::NEG_INFINITY;
        let mut best_pair = (0, 0);

        for (ai, agent) in agents.iter().enumerate() {
            if assigned_agents[ai] {
                continue;
            }
            for (ti, task) in tasks.iter().enumerate() {
                if assigned_tasks[ti] {
                    continue;
                }
                let d = agent.position.distance_to(task.position).max(0.01);
                let bid = task.priority / d;
                if bid > best_bid {
                    best_bid = bid;
                    best_pair = (ai, ti);
                }
            }
        }

        if best_bid > f64::NEG_INFINITY {
            let (ai, ti) = best_pair;
            allocation.insert(agents[ai].id, tasks[ti].id);
            assigned_agents[ai] = true;
            assigned_tasks[ti] = true;
        }
    }

    allocation
}
