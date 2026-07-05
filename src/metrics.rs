//! Swarm metrics: centroid / speed / spread / order / distance / connectivity.

use crate::agent::Agent;
use crate::vec2::Vec2;

/// Compute the centroid (center of mass) of the swarm.
#[must_use]
pub fn swarm_centroid(agents: &[Agent]) -> Vec2 {
    if agents.is_empty() {
        return Vec2::zero();
    }
    let mut sum = Vec2::zero();
    for a in agents {
        sum += a.position;
    }
    #[allow(clippy::cast_precision_loss)]
    let n = agents.len() as f64;
    sum * (1.0 / n)
}

/// Compute the average speed of the swarm.
#[must_use]
pub fn swarm_avg_speed(agents: &[Agent]) -> f64 {
    if agents.is_empty() {
        return 0.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let n = agents.len() as f64;
    agents.iter().map(|a| a.velocity.length()).sum::<f64>() / n
}

/// Compute the spread (standard deviation of distance from centroid) of the swarm.
#[must_use]
pub fn swarm_spread(agents: &[Agent]) -> f64 {
    if agents.is_empty() {
        return 0.0;
    }
    let c = swarm_centroid(agents);
    #[allow(clippy::cast_precision_loss)]
    let n = agents.len() as f64;
    let variance = agents
        .iter()
        .map(|a| {
            let d = a.position.distance_to(c);
            d * d
        })
        .sum::<f64>()
        / n;
    variance.sqrt()
}

/// Compute the velocity alignment (order parameter).
#[must_use]
pub fn swarm_order(agents: &[Agent]) -> f64 {
    if agents.is_empty() {
        return 0.0;
    }
    let mut sum_vel = Vec2::zero();
    let mut sum_speed = 0.0;
    for a in agents {
        sum_vel += a.velocity;
        sum_speed += a.velocity.length();
    }
    if sum_speed < 1e-12 {
        return 0.0;
    }
    sum_vel.length() / sum_speed
}

/// Compute minimum pairwise distance in the swarm.
#[must_use]
pub fn swarm_min_distance(agents: &[Agent]) -> f64 {
    let mut min_d: Option<f64> = None;
    for i in 0..agents.len() {
        for j in (i + 1)..agents.len() {
            let d = agents[i].position.distance_to(agents[j].position);
            min_d = Some(min_d.map_or(d, |m: f64| m.min(d)));
        }
    }
    min_d.unwrap_or(0.0)
}

/// Compute the maximum pairwise distance (diameter) in the swarm.
#[must_use]
pub fn swarm_diameter(agents: &[Agent]) -> f64 {
    let mut max_d: f64 = 0.0;
    for i in 0..agents.len() {
        for j in (i + 1)..agents.len() {
            let d = agents[i].position.distance_to(agents[j].position);
            if d > max_d {
                max_d = d;
            }
        }
    }
    max_d
}

/// Count the number of collisions (pairs closer than `threshold`).
#[must_use]
pub fn swarm_collision_count(agents: &[Agent], threshold: f64) -> usize {
    let mut count = 0;
    for i in 0..agents.len() {
        for j in (i + 1)..agents.len() {
            if agents[i].position.distance_to(agents[j].position) < threshold {
                count += 1;
            }
        }
    }
    count
}

/// Compute connectivity ratio: fraction of agent pairs within communication range.
#[must_use]
pub fn swarm_connectivity(agents: &[Agent], comm_range: f64) -> f64 {
    let n = agents.len();
    if n < 2 {
        return 1.0;
    }
    let mut connected = 0u64;
    let total_pairs = n * (n - 1) / 2;
    for i in 0..n {
        for j in (i + 1)..n {
            if agents[i].position.distance_to(agents[j].position) <= comm_range {
                connected += 1;
            }
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let ratio = connected as f64 / total_pairs as f64;
    ratio
}
