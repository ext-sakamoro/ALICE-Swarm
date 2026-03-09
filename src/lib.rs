#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::module_name_repetitions)]

//! ALICE-Swarm: Multi-agent swarm control library.
//!
//! Provides Boids algorithm, formation control, consensus protocols,
//! task allocation, obstacle avoidance, communication topology,
//! leader-follower dynamics, and swarm metrics.

use core::f64;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Vec2
// ---------------------------------------------------------------------------

/// 2D vector for swarm computations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    #[must_use]
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    #[must_use]
    pub const fn zero() -> Self {
        Self { x: 0.0, y: 0.0 }
    }

    #[must_use]
    pub fn length(self) -> f64 {
        self.x.hypot(self.y)
    }

    #[must_use]
    pub fn length_sq(self) -> f64 {
        self.x.mul_add(self.x, self.y * self.y)
    }

    #[must_use]
    pub fn normalized(self) -> Self {
        let len = self.length();
        if len < 1e-12 {
            Self::zero()
        } else {
            Self {
                x: self.x / len,
                y: self.y / len,
            }
        }
    }

    #[must_use]
    pub fn distance_to(self, other: Self) -> f64 {
        (self - other).length()
    }

    #[must_use]
    pub fn dot(self, other: Self) -> f64 {
        self.x.mul_add(other.x, self.y * other.y)
    }

    #[must_use]
    pub fn clamped(self, max_len: f64) -> Self {
        let len = self.length();
        if len > max_len && len > 1e-12 {
            let scale = max_len / len;
            Self {
                x: self.x * scale,
                y: self.y * scale,
            }
        } else {
            self
        }
    }
}

impl std::ops::Add for Vec2 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl std::ops::Sub for Vec2 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl std::ops::Mul<f64> for Vec2 {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}

impl std::ops::AddAssign for Vec2 {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

// ---------------------------------------------------------------------------
// Agent
// ---------------------------------------------------------------------------

/// A single swarm agent with position, velocity, and an optional leader flag.
#[derive(Debug, Clone)]
pub struct Agent {
    pub id: usize,
    pub position: Vec2,
    pub velocity: Vec2,
    pub is_leader: bool,
}

impl Agent {
    #[must_use]
    pub const fn new(id: usize, position: Vec2, velocity: Vec2) -> Self {
        Self {
            id,
            position,
            velocity,
            is_leader: false,
        }
    }

    #[must_use]
    pub const fn with_leader(mut self, leader: bool) -> Self {
        self.is_leader = leader;
        self
    }
}

// ---------------------------------------------------------------------------
// Obstacle
// ---------------------------------------------------------------------------

/// A circular obstacle in the environment.
#[derive(Debug, Clone, Copy)]
pub struct Obstacle {
    pub center: Vec2,
    pub radius: f64,
}

impl Obstacle {
    #[must_use]
    pub const fn new(center: Vec2, radius: f64) -> Self {
        Self { center, radius }
    }
}

// ---------------------------------------------------------------------------
// BoidParams
// ---------------------------------------------------------------------------

/// Parameters for the Boids algorithm.
#[derive(Debug, Clone, Copy)]
pub struct BoidParams {
    pub separation_radius: f64,
    pub alignment_radius: f64,
    pub cohesion_radius: f64,
    pub separation_weight: f64,
    pub alignment_weight: f64,
    pub cohesion_weight: f64,
    pub max_speed: f64,
    pub max_force: f64,
}

impl Default for BoidParams {
    fn default() -> Self {
        Self {
            separation_radius: 2.0,
            alignment_radius: 5.0,
            cohesion_radius: 5.0,
            separation_weight: 1.5,
            alignment_weight: 1.0,
            cohesion_weight: 1.0,
            max_speed: 4.0,
            max_force: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Boids algorithm
// ---------------------------------------------------------------------------

/// Compute the separation steering vector for agent at `index`.
#[must_use]
pub fn boids_separation(agents: &[Agent], index: usize, radius: f64) -> Vec2 {
    let me = &agents[index];
    let mut steer = Vec2::zero();
    let mut count = 0u32;
    for (i, other) in agents.iter().enumerate() {
        if i == index {
            continue;
        }
        let d = me.position.distance_to(other.position);
        if d < radius && d > 1e-12 {
            let diff = (me.position - other.position).normalized() * (1.0 / d);
            steer += diff;
            count += 1;
        }
    }
    if count > 0 {
        steer = steer * (1.0 / f64::from(count));
    }
    steer
}

/// Compute the alignment steering vector for agent at `index`.
#[must_use]
pub fn boids_alignment(agents: &[Agent], index: usize, radius: f64) -> Vec2 {
    let me = &agents[index];
    let mut avg_vel = Vec2::zero();
    let mut count = 0u32;
    for (i, other) in agents.iter().enumerate() {
        if i == index {
            continue;
        }
        let d = me.position.distance_to(other.position);
        if d < radius {
            avg_vel += other.velocity;
            count += 1;
        }
    }
    if count > 0 {
        avg_vel = avg_vel * (1.0 / f64::from(count));
        avg_vel - me.velocity
    } else {
        Vec2::zero()
    }
}

/// Compute the cohesion steering vector for agent at `index`.
#[must_use]
pub fn boids_cohesion(agents: &[Agent], index: usize, radius: f64) -> Vec2 {
    let me = &agents[index];
    let mut center = Vec2::zero();
    let mut count = 0u32;
    for (i, other) in agents.iter().enumerate() {
        if i == index {
            continue;
        }
        let d = me.position.distance_to(other.position);
        if d < radius {
            center += other.position;
            count += 1;
        }
    }
    if count > 0 {
        center = center * (1.0 / f64::from(count));
        center - me.position
    } else {
        Vec2::zero()
    }
}

/// Compute the combined Boids steering for a single agent. Returns the desired acceleration.
#[must_use]
pub fn boids_steer(agents: &[Agent], index: usize, params: &BoidParams) -> Vec2 {
    let sep = boids_separation(agents, index, params.separation_radius) * params.separation_weight;
    let ali = boids_alignment(agents, index, params.alignment_radius) * params.alignment_weight;
    let coh = boids_cohesion(agents, index, params.cohesion_radius) * params.cohesion_weight;
    (sep + ali + coh).clamped(params.max_force)
}

/// Advance the swarm by one time step using the Boids algorithm.
pub fn boids_step(agents: &mut [Agent], params: &BoidParams, dt: f64) {
    let steers: Vec<Vec2> = (0..agents.len())
        .map(|i| boids_steer(agents, i, params))
        .collect();

    for (agent, &steer) in agents.iter_mut().zip(steers.iter()) {
        agent.velocity = (agent.velocity + steer * dt).clamped(params.max_speed);
        agent.position += agent.velocity * dt;
    }
}

// ---------------------------------------------------------------------------
// Formation control
// ---------------------------------------------------------------------------

/// Formation shape definition: each slot is an offset from the formation center.
#[derive(Debug, Clone)]
pub struct Formation {
    pub offsets: Vec<Vec2>,
}

impl Formation {
    /// Create a line formation along the X-axis with given spacing.
    #[must_use]
    pub fn line(count: usize, spacing: f64) -> Self {
        let offsets = (0..count)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let x = (i as f64 - (count as f64 - 1.0) / 2.0) * spacing;
                Vec2::new(x, 0.0)
            })
            .collect();
        Self { offsets }
    }

    /// Create a ring formation with given radius.
    #[must_use]
    pub fn ring(count: usize, radius: f64) -> Self {
        let offsets = (0..count)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let angle = 2.0 * f64::consts::PI * (i as f64) / (count as f64);
                Vec2::new(radius * angle.cos(), radius * angle.sin())
            })
            .collect();
        Self { offsets }
    }

    /// Create a V-formation (wedge) with given spacing and opening angle.
    #[must_use]
    pub fn v_shape(count: usize, spacing: f64, half_angle: f64) -> Self {
        let mut offsets = Vec::with_capacity(count);
        offsets.push(Vec2::zero()); // leader at origin
        for i in 1..count {
            #[allow(clippy::cast_precision_loss)]
            let rank = i.div_ceil(2) as f64;
            let side = if i % 2 == 1 { 1.0 } else { -1.0 };
            let x = side * rank * spacing * half_angle.sin();
            let y = -rank * spacing * half_angle.cos();
            offsets.push(Vec2::new(x, y));
        }
        Self { offsets }
    }

    /// Create a grid formation with given rows, cols, and spacing.
    #[must_use]
    pub fn grid(rows: usize, cols: usize, spacing: f64) -> Self {
        let mut offsets = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                #[allow(clippy::cast_precision_loss)]
                let x = (c as f64 - (cols as f64 - 1.0) / 2.0) * spacing;
                #[allow(clippy::cast_precision_loss)]
                let y = (r as f64 - (rows as f64 - 1.0) / 2.0) * spacing;
                offsets.push(Vec2::new(x, y));
            }
        }
        Self { offsets }
    }

    /// Number of slots in this formation.
    #[must_use]
    pub const fn slot_count(&self) -> usize {
        self.offsets.len()
    }
}

/// Compute steering forces to move agents toward their assigned formation slots.
///
/// `center` is the world-space center of the formation.
/// Returns one force vector per agent (up to `min(agents.len(), formation.slot_count())`).
#[must_use]
pub fn formation_steer(
    agents: &[Agent],
    formation: &Formation,
    center: Vec2,
    gain: f64,
) -> Vec<Vec2> {
    let n = agents.len().min(formation.slot_count());
    (0..n)
        .map(|i| {
            let target = center + formation.offsets[i];
            (target - agents[i].position) * gain
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Consensus protocol (average consensus)
// ---------------------------------------------------------------------------

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
        // hub connects to all
        neighbors.push((1..n).collect());
        // spokes connect to hub only
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
/// Each agent updates its value toward the average of its neighbors.
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

// ---------------------------------------------------------------------------
// Task allocation
// ---------------------------------------------------------------------------

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

    // Sort tasks by descending priority
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
/// Each agent bids on each task (bid = priority / distance). Highest bidder wins.
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

// ---------------------------------------------------------------------------
// Obstacle avoidance
// ---------------------------------------------------------------------------

/// Compute repulsive force from obstacles for a given agent.
#[must_use]
pub fn obstacle_avoidance(
    agent: &Agent,
    obstacles: &[Obstacle],
    avoidance_radius: f64,
    strength: f64,
) -> Vec2 {
    let mut force = Vec2::zero();
    for obs in obstacles {
        let to_agent = agent.position - obs.center;
        let dist = to_agent.length() - obs.radius;
        if dist < avoidance_radius && dist > 1e-12 {
            let repulsion = to_agent.normalized() * (strength / (dist * dist));
            force += repulsion;
        }
    }
    force
}

/// Check if a straight-line path between two points intersects any obstacle.
#[must_use]
pub fn path_blocked(from: Vec2, to: Vec2, obstacles: &[Obstacle]) -> bool {
    let dir = to - from;
    let len = dir.length();
    if len < 1e-12 {
        return false;
    }
    let unit_dir = dir.normalized();

    for obs in obstacles {
        let offset = from - obs.center;
        let coeff_a = unit_dir.dot(unit_dir);
        let coeff_b = 2.0 * offset.dot(unit_dir);
        let coeff_c = obs.radius.mul_add(-obs.radius, offset.dot(offset));
        let discriminant = coeff_b.mul_add(coeff_b, -4.0 * coeff_a * coeff_c);
        if discriminant >= 0.0 {
            let sqrt_disc = discriminant.sqrt();
            let t1 = (-coeff_b - sqrt_disc) / (2.0 * coeff_a);
            let t2 = (-coeff_b + sqrt_disc) / (2.0 * coeff_a);
            if t1 <= len && t2 >= 0.0 {
                return true;
            }
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Leader-follower
// ---------------------------------------------------------------------------

/// Compute follower steering to track a leader agent.
/// The follower aims for a point behind the leader (offset along negative velocity direction).
#[must_use]
pub fn leader_follower_steer(
    follower: &Agent,
    leader: &Agent,
    follow_distance: f64,
    gain: f64,
) -> Vec2 {
    let leader_dir = if leader.velocity.length_sq() > 1e-12 {
        leader.velocity.normalized()
    } else {
        Vec2::new(1.0, 0.0)
    };
    let target = leader.position - leader_dir * follow_distance;
    (target - follower.position) * gain
}

/// Advance a leader-follower swarm by one step.
/// The leader moves according to `leader_velocity`. Followers track the agent
/// designated as their leader via simple proportional control.
pub fn leader_follower_step(
    agents: &mut [Agent],
    leader_velocity: Vec2,
    follow_distance: f64,
    gain: f64,
    dt: f64,
) {
    if agents.is_empty() {
        return;
    }
    // Agent 0 is the leader
    agents[0].velocity = leader_velocity;
    agents[0].position += leader_velocity * dt;

    // Each subsequent agent follows the one before it
    let positions: Vec<Vec2> = agents.iter().map(|a| a.position).collect();
    let velocities: Vec<Vec2> = agents.iter().map(|a| a.velocity).collect();

    for i in 1..agents.len() {
        let leader_agent = Agent {
            id: i - 1,
            position: positions[i - 1],
            velocity: velocities[i - 1],
            is_leader: false,
        };
        let steer = leader_follower_steer(&agents[i], &leader_agent, follow_distance, gain);
        agents[i].velocity =
            (agents[i].velocity + steer * dt).clamped(leader_velocity.length() * 1.5);
        agents[i].position += agents[i].velocity * dt;
    }
}

// ---------------------------------------------------------------------------
// Swarm metrics
// ---------------------------------------------------------------------------

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

/// Compute the velocity alignment (order parameter). 1.0 = perfectly aligned, 0.0 = random.
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6
    }

    fn make_agents(positions: &[(f64, f64)]) -> Vec<Agent> {
        positions
            .iter()
            .enumerate()
            .map(|(i, &(x, y))| Agent::new(i, Vec2::new(x, y), Vec2::zero()))
            .collect()
    }

    fn make_agents_with_vel(data: &[(f64, f64, f64, f64)]) -> Vec<Agent> {
        data.iter()
            .enumerate()
            .map(|(i, &(px, py, vx, vy))| Agent::new(i, Vec2::new(px, py), Vec2::new(vx, vy)))
            .collect()
    }

    // --- Vec2 tests ---

    #[test]
    fn test_vec2_zero() {
        let v = Vec2::zero();
        assert!(approx_eq(v.x, 0.0));
        assert!(approx_eq(v.y, 0.0));
    }

    #[test]
    fn test_vec2_new() {
        let v = Vec2::new(3.0, 4.0);
        assert!(approx_eq(v.x, 3.0));
        assert!(approx_eq(v.y, 4.0));
    }

    #[test]
    fn test_vec2_length() {
        let v = Vec2::new(3.0, 4.0);
        assert!(approx_eq(v.length(), 5.0));
    }

    #[test]
    fn test_vec2_length_sq() {
        let v = Vec2::new(3.0, 4.0);
        assert!(approx_eq(v.length_sq(), 25.0));
    }

    #[test]
    fn test_vec2_normalized() {
        let v = Vec2::new(0.0, 5.0).normalized();
        assert!(approx_eq(v.x, 0.0));
        assert!(approx_eq(v.y, 1.0));
    }

    #[test]
    fn test_vec2_normalized_zero() {
        let v = Vec2::zero().normalized();
        assert!(approx_eq(v.length(), 0.0));
    }

    #[test]
    fn test_vec2_distance() {
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(3.0, 4.0);
        assert!(approx_eq(a.distance_to(b), 5.0));
    }

    #[test]
    fn test_vec2_dot() {
        let a = Vec2::new(1.0, 2.0);
        let b = Vec2::new(3.0, 4.0);
        assert!(approx_eq(a.dot(b), 11.0));
    }

    #[test]
    fn test_vec2_add() {
        let r = Vec2::new(1.0, 2.0) + Vec2::new(3.0, 4.0);
        assert!(approx_eq(r.x, 4.0));
        assert!(approx_eq(r.y, 6.0));
    }

    #[test]
    fn test_vec2_sub() {
        let r = Vec2::new(5.0, 7.0) - Vec2::new(2.0, 3.0);
        assert!(approx_eq(r.x, 3.0));
        assert!(approx_eq(r.y, 4.0));
    }

    #[test]
    fn test_vec2_mul() {
        let r = Vec2::new(2.0, 3.0) * 2.0;
        assert!(approx_eq(r.x, 4.0));
        assert!(approx_eq(r.y, 6.0));
    }

    #[test]
    fn test_vec2_add_assign() {
        let mut v = Vec2::new(1.0, 2.0);
        v += Vec2::new(3.0, 4.0);
        assert!(approx_eq(v.x, 4.0));
        assert!(approx_eq(v.y, 6.0));
    }

    #[test]
    fn test_vec2_clamped_within() {
        let v = Vec2::new(1.0, 0.0);
        let c = v.clamped(5.0);
        assert!(approx_eq(c.length(), 1.0));
    }

    #[test]
    fn test_vec2_clamped_over() {
        let v = Vec2::new(10.0, 0.0);
        let c = v.clamped(3.0);
        assert!(approx_eq(c.length(), 3.0));
    }

    // --- Agent tests ---

    #[test]
    fn test_agent_new() {
        let a = Agent::new(42, Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
        assert_eq!(a.id, 42);
        assert!(!a.is_leader);
    }

    #[test]
    fn test_agent_with_leader() {
        let a = Agent::new(0, Vec2::zero(), Vec2::zero()).with_leader(true);
        assert!(a.is_leader);
    }

    // --- Boids separation ---

    #[test]
    fn test_boids_separation_no_neighbors() {
        let agents = make_agents(&[(0.0, 0.0), (100.0, 100.0)]);
        let s = boids_separation(&agents, 0, 2.0);
        assert!(approx_eq(s.length(), 0.0));
    }

    #[test]
    fn test_boids_separation_close_neighbor() {
        let agents = make_agents(&[(0.0, 0.0), (1.0, 0.0)]);
        let s = boids_separation(&agents, 0, 5.0);
        assert!(s.x < 0.0); // pushed away to the left
    }

    #[test]
    fn test_boids_separation_symmetry() {
        let agents = make_agents(&[(0.0, 0.0), (1.0, 0.0)]);
        let s0 = boids_separation(&agents, 0, 5.0);
        let s1 = boids_separation(&agents, 1, 5.0);
        // opposite directions
        assert!(approx_eq(s0.x + s1.x, 0.0));
    }

    // --- Boids alignment ---

    #[test]
    fn test_boids_alignment_same_velocity() {
        let agents = make_agents_with_vel(&[(0.0, 0.0, 1.0, 0.0), (1.0, 0.0, 1.0, 0.0)]);
        let a = boids_alignment(&agents, 0, 5.0);
        assert!(approx_eq(a.length(), 0.0));
    }

    #[test]
    fn test_boids_alignment_different_velocity() {
        let agents = make_agents_with_vel(&[(0.0, 0.0, 1.0, 0.0), (1.0, 0.0, -1.0, 0.0)]);
        let a = boids_alignment(&agents, 0, 5.0);
        assert!(a.x < 0.0); // should steer toward neighbor's velocity direction
    }

    #[test]
    fn test_boids_alignment_no_neighbors() {
        let agents = make_agents_with_vel(&[(0.0, 0.0, 1.0, 0.0), (100.0, 0.0, -1.0, 0.0)]);
        let a = boids_alignment(&agents, 0, 5.0);
        assert!(approx_eq(a.length(), 0.0));
    }

    // --- Boids cohesion ---

    #[test]
    fn test_boids_cohesion_toward_group() {
        let agents = make_agents(&[(0.0, 0.0), (2.0, 0.0), (2.0, 2.0)]);
        let c = boids_cohesion(&agents, 0, 10.0);
        assert!(c.x > 0.0); // should steer right
        assert!(c.y > 0.0); // should steer up
    }

    #[test]
    fn test_boids_cohesion_no_neighbors() {
        let agents = make_agents(&[(0.0, 0.0), (100.0, 100.0)]);
        let c = boids_cohesion(&agents, 0, 5.0);
        assert!(approx_eq(c.length(), 0.0));
    }

    // --- Boids steer ---

    #[test]
    fn test_boids_steer_returns_clamped() {
        let agents = make_agents(&[(0.0, 0.0), (0.5, 0.0), (0.0, 0.5)]);
        let params = BoidParams::default();
        let s = boids_steer(&agents, 0, &params);
        assert!(s.length() <= params.max_force + EPS);
    }

    // --- Boids step ---

    #[test]
    fn test_boids_step_moves_agents() {
        let mut agents = make_agents_with_vel(&[
            (0.0, 0.0, 1.0, 0.0),
            (1.0, 0.0, 1.0, 0.0),
            (0.5, 1.0, 0.0, -1.0),
        ]);
        let params = BoidParams::default();
        let old_pos: Vec<Vec2> = agents.iter().map(|a| a.position).collect();
        boids_step(&mut agents, &params, 0.1);
        let changed = agents
            .iter()
            .zip(old_pos.iter())
            .any(|(a, &op)| a.position.distance_to(op) > EPS);
        assert!(changed);
    }

    #[test]
    fn test_boids_step_respects_max_speed() {
        let mut agents = make_agents_with_vel(&[(0.0, 0.0, 3.0, 0.0), (0.1, 0.0, 3.0, 0.0)]);
        let params = BoidParams {
            max_speed: 2.0,
            ..BoidParams::default()
        };
        boids_step(&mut agents, &params, 1.0);
        for a in &agents {
            assert!(a.velocity.length() <= params.max_speed + EPS);
        }
    }

    #[test]
    fn test_boids_step_multiple_iterations() {
        let mut agents = make_agents_with_vel(&[
            (0.0, 0.0, 1.0, 0.5),
            (3.0, 0.0, -1.0, 0.5),
            (1.5, 3.0, 0.0, -1.0),
        ]);
        let params = BoidParams::default();
        for _ in 0..50 {
            boids_step(&mut agents, &params, 0.05);
        }
        // Should still have finite positions
        for a in &agents {
            assert!(a.position.length() < 1000.0);
        }
    }

    // --- Formation tests ---

    #[test]
    fn test_formation_line() {
        let f = Formation::line(5, 2.0);
        assert_eq!(f.slot_count(), 5);
        // Should be centered around 0
        let sum: f64 = f.offsets.iter().map(|o| o.x).sum();
        assert!(approx_eq(sum, 0.0));
    }

    #[test]
    fn test_formation_ring() {
        let f = Formation::ring(4, 10.0);
        assert_eq!(f.slot_count(), 4);
        // All points should be at distance 10 from origin
        for o in &f.offsets {
            assert!(approx_eq(o.length(), 10.0));
        }
    }

    #[test]
    fn test_formation_v_shape() {
        let f = Formation::v_shape(5, 2.0, f64::consts::FRAC_PI_4);
        assert_eq!(f.slot_count(), 5);
        // First agent (leader) at origin
        assert!(approx_eq(f.offsets[0].length(), 0.0));
    }

    #[test]
    fn test_formation_grid() {
        let f = Formation::grid(3, 4, 1.0);
        assert_eq!(f.slot_count(), 12);
    }

    #[test]
    fn test_formation_steer_toward_target() {
        let agents = make_agents(&[(0.0, 0.0), (0.0, 0.0)]);
        let f = Formation::line(2, 2.0);
        let center = Vec2::new(10.0, 0.0);
        let steers = formation_steer(&agents, &f, center, 1.0);
        assert_eq!(steers.len(), 2);
        // Both should steer toward formation slots near x=10
        for s in &steers {
            assert!(s.x > 0.0);
        }
    }

    #[test]
    fn test_formation_steer_at_position() {
        let f = Formation::line(1, 1.0);
        let center = Vec2::new(5.0, 0.0);
        let agents = vec![Agent::new(0, center + f.offsets[0], Vec2::zero())];
        let steers = formation_steer(&agents, &f, center, 1.0);
        assert!(approx_eq(steers[0].length(), 0.0));
    }

    // --- Topology tests ---

    #[test]
    fn test_topology_fully_connected() {
        let t = Topology::fully_connected(4);
        assert_eq!(t.agent_count(), 4);
        for i in 0..4 {
            assert_eq!(t.neighbors[i].len(), 3);
        }
    }

    #[test]
    fn test_topology_ring() {
        let t = Topology::ring(5);
        assert_eq!(t.agent_count(), 5);
        for i in 0..5 {
            assert_eq!(t.neighbors[i].len(), 2);
        }
        assert!(t.is_connected(0, 1));
        assert!(t.is_connected(0, 4));
        assert!(!t.is_connected(0, 2));
    }

    #[test]
    fn test_topology_ring_single() {
        let t = Topology::ring(1);
        assert_eq!(t.neighbors[0].len(), 0);
    }

    #[test]
    fn test_topology_ring_empty() {
        let t = Topology::ring(0);
        assert_eq!(t.agent_count(), 0);
    }

    #[test]
    fn test_topology_star() {
        let t = Topology::star(5);
        assert_eq!(t.neighbors[0].len(), 4); // hub
        for i in 1..5 {
            assert_eq!(t.neighbors[i].len(), 1); // spokes
            assert!(t.is_connected(i, 0));
        }
    }

    #[test]
    fn test_topology_star_empty() {
        let t = Topology::star(0);
        assert_eq!(t.agent_count(), 0);
    }

    #[test]
    fn test_topology_k_nearest() {
        let agents = make_agents(&[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (10.0, 0.0)]);
        let t = Topology::k_nearest(&agents, 2);
        assert!(t.is_connected(0, 1));
        assert!(t.is_connected(0, 2));
        assert!(!t.is_connected(0, 3));
    }

    // --- Consensus tests ---

    #[test]
    fn test_consensus_converges() {
        let topo = Topology::fully_connected(4);
        let mut values = vec![0.0, 4.0, 8.0, 12.0];
        for _ in 0..100 {
            consensus_step(&mut values, &topo, 0.5);
        }
        let avg = 6.0;
        for &v in &values {
            assert!(approx_eq(v, avg));
        }
    }

    #[test]
    fn test_consensus_ring_converges() {
        let topo = Topology::ring(3);
        let mut values = vec![0.0, 3.0, 6.0];
        for _ in 0..200 {
            consensus_step(&mut values, &topo, 0.3);
        }
        let avg = 3.0;
        for &v in &values {
            assert!((v - avg).abs() < 0.01);
        }
    }

    #[test]
    fn test_consensus_vec2_converges() {
        let topo = Topology::fully_connected(3);
        let mut values = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(3.0, 0.0),
            Vec2::new(0.0, 3.0),
        ];
        for _ in 0..100 {
            consensus_step_vec2(&mut values, &topo, 0.5);
        }
        for v in &values {
            assert!(approx_eq(v.x, 1.0));
            assert!(approx_eq(v.y, 1.0));
        }
    }

    #[test]
    fn test_consensus_single_agent() {
        let topo = Topology::fully_connected(1);
        let mut values = vec![42.0];
        consensus_step(&mut values, &topo, 0.5);
        assert!(approx_eq(values[0], 42.0));
    }

    // --- Task allocation tests ---

    #[test]
    fn test_greedy_allocation() {
        let agents = make_agents(&[(0.0, 0.0), (10.0, 0.0)]);
        let tasks = vec![
            Task::new(0, Vec2::new(1.0, 0.0), 1.0),
            Task::new(1, Vec2::new(9.0, 0.0), 1.0),
        ];
        let alloc = allocate_greedy(&agents, &tasks);
        assert_eq!(alloc[&0], 0); // agent 0 -> task 0
        assert_eq!(alloc[&1], 1); // agent 1 -> task 1
    }

    #[test]
    fn test_greedy_allocation_priority() {
        let agents = make_agents(&[(5.0, 0.0)]);
        let tasks = vec![
            Task::new(0, Vec2::new(0.0, 0.0), 1.0),
            Task::new(1, Vec2::new(10.0, 0.0), 10.0),
        ];
        let alloc = allocate_greedy(&agents, &tasks);
        // Higher priority task 1 should be allocated first
        assert_eq!(alloc[&0], 1);
    }

    #[test]
    fn test_auction_allocation() {
        let agents = make_agents(&[(0.0, 0.0), (10.0, 0.0)]);
        let tasks = vec![
            Task::new(0, Vec2::new(0.5, 0.0), 1.0),
            Task::new(1, Vec2::new(9.5, 0.0), 1.0),
        ];
        let alloc = allocate_auction(&agents, &tasks);
        assert_eq!(alloc[&0], 0);
        assert_eq!(alloc[&1], 1);
    }

    #[test]
    fn test_allocation_more_agents_than_tasks() {
        let agents = make_agents(&[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]);
        let tasks = vec![Task::new(0, Vec2::new(0.0, 0.0), 1.0)];
        let alloc = allocate_greedy(&agents, &tasks);
        assert_eq!(alloc.len(), 1);
    }

    #[test]
    fn test_allocation_more_tasks_than_agents() {
        let agents = make_agents(&[(0.0, 0.0)]);
        let tasks = vec![
            Task::new(0, Vec2::new(0.0, 0.0), 1.0),
            Task::new(1, Vec2::new(1.0, 0.0), 1.0),
        ];
        let alloc = allocate_greedy(&agents, &tasks);
        assert_eq!(alloc.len(), 1);
    }

    #[test]
    fn test_auction_empty() {
        let agents: Vec<Agent> = vec![];
        let tasks: Vec<Task> = vec![];
        let alloc = allocate_auction(&agents, &tasks);
        assert!(alloc.is_empty());
    }

    // --- Obstacle avoidance tests ---

    #[test]
    fn test_obstacle_avoidance_repels() {
        let agent = Agent::new(0, Vec2::new(2.0, 0.0), Vec2::new(0.0, 0.0));
        let obstacles = vec![Obstacle::new(Vec2::new(0.0, 0.0), 1.0)];
        let force = obstacle_avoidance(&agent, &obstacles, 5.0, 1.0);
        assert!(force.x > 0.0); // pushed away from obstacle
    }

    #[test]
    fn test_obstacle_avoidance_no_effect_far() {
        let agent = Agent::new(0, Vec2::new(100.0, 0.0), Vec2::zero());
        let obstacles = vec![Obstacle::new(Vec2::zero(), 1.0)];
        let force = obstacle_avoidance(&agent, &obstacles, 5.0, 1.0);
        assert!(approx_eq(force.length(), 0.0));
    }

    #[test]
    fn test_obstacle_avoidance_multiple() {
        let agent = Agent::new(0, Vec2::new(0.0, 0.0), Vec2::zero());
        let obstacles = vec![
            Obstacle::new(Vec2::new(-2.0, 0.0), 0.5),
            Obstacle::new(Vec2::new(2.0, 0.0), 0.5),
        ];
        let force = obstacle_avoidance(&agent, &obstacles, 5.0, 1.0);
        // Symmetric -> x component cancels
        assert!(force.x.abs() < 0.01);
    }

    #[test]
    fn test_path_blocked_true() {
        let obstacles = vec![Obstacle::new(Vec2::new(5.0, 0.0), 1.0)];
        assert!(path_blocked(
            Vec2::new(0.0, 0.0),
            Vec2::new(10.0, 0.0),
            &obstacles
        ));
    }

    #[test]
    fn test_path_blocked_false() {
        let obstacles = vec![Obstacle::new(Vec2::new(5.0, 5.0), 1.0)];
        assert!(!path_blocked(
            Vec2::new(0.0, 0.0),
            Vec2::new(10.0, 0.0),
            &obstacles
        ));
    }

    #[test]
    fn test_path_blocked_no_obstacles() {
        assert!(!path_blocked(
            Vec2::new(0.0, 0.0),
            Vec2::new(10.0, 0.0),
            &[]
        ));
    }

    #[test]
    fn test_path_blocked_zero_length() {
        let obstacles = vec![Obstacle::new(Vec2::new(0.0, 0.0), 1.0)];
        assert!(!path_blocked(
            Vec2::new(5.0, 5.0),
            Vec2::new(5.0, 5.0),
            &obstacles
        ));
    }

    // --- Leader-follower tests ---

    #[test]
    fn test_leader_follower_steer_behind() {
        let leader = Agent::new(0, Vec2::new(10.0, 0.0), Vec2::new(1.0, 0.0));
        let follower = Agent::new(1, Vec2::new(0.0, 0.0), Vec2::zero());
        let steer = leader_follower_steer(&follower, &leader, 2.0, 1.0);
        assert!(steer.x > 0.0); // follower should steer toward leader's tail
    }

    #[test]
    fn test_leader_follower_steer_at_target() {
        let leader = Agent::new(0, Vec2::new(5.0, 0.0), Vec2::new(1.0, 0.0));
        let follower = Agent::new(1, Vec2::new(3.0, 0.0), Vec2::zero());
        let steer = leader_follower_steer(&follower, &leader, 2.0, 1.0);
        // target is at (3.0, 0.0), follower is at (3.0, 0.0) -> zero steer
        assert!(approx_eq(steer.length(), 0.0));
    }

    #[test]
    fn test_leader_follower_step() {
        let mut agents = vec![
            Agent::new(0, Vec2::new(0.0, 0.0), Vec2::zero()).with_leader(true),
            Agent::new(1, Vec2::new(-3.0, 0.0), Vec2::zero()),
            Agent::new(2, Vec2::new(-6.0, 0.0), Vec2::zero()),
        ];
        let leader_vel = Vec2::new(2.0, 0.0);
        for _ in 0..50 {
            leader_follower_step(&mut agents, leader_vel, 2.0, 1.0, 0.1);
        }
        // Leader should have moved forward
        assert!(agents[0].position.x > 5.0);
        // Followers should be trailing
        assert!(agents[1].position.x < agents[0].position.x);
        assert!(agents[2].position.x < agents[1].position.x);
    }

    #[test]
    fn test_leader_follower_step_empty() {
        let mut agents: Vec<Agent> = vec![];
        leader_follower_step(&mut agents, Vec2::new(1.0, 0.0), 2.0, 1.0, 0.1);
        assert!(agents.is_empty());
    }

    // --- Swarm metrics tests ---

    #[test]
    fn test_centroid() {
        let agents = make_agents(&[(0.0, 0.0), (4.0, 0.0), (0.0, 4.0)]);
        let c = swarm_centroid(&agents);
        assert!(approx_eq(c.x, 4.0 / 3.0));
        assert!(approx_eq(c.y, 4.0 / 3.0));
    }

    #[test]
    fn test_centroid_empty() {
        let c = swarm_centroid(&[]);
        assert!(approx_eq(c.length(), 0.0));
    }

    #[test]
    fn test_avg_speed() {
        let agents = make_agents_with_vel(&[(0.0, 0.0, 3.0, 4.0), (0.0, 0.0, 0.0, 0.0)]);
        let s = swarm_avg_speed(&agents);
        assert!(approx_eq(s, 2.5)); // (5+0)/2
    }

    #[test]
    fn test_avg_speed_empty() {
        assert!(approx_eq(swarm_avg_speed(&[]), 0.0));
    }

    #[test]
    fn test_spread() {
        let agents = make_agents(&[(0.0, 0.0), (2.0, 0.0)]);
        let s = swarm_spread(&agents);
        assert!(approx_eq(s, 1.0)); // both 1.0 from centroid
    }

    #[test]
    fn test_spread_single() {
        let agents = make_agents(&[(5.0, 5.0)]);
        assert!(approx_eq(swarm_spread(&agents), 0.0));
    }

    #[test]
    fn test_spread_empty() {
        assert!(approx_eq(swarm_spread(&[]), 0.0));
    }

    #[test]
    fn test_order_aligned() {
        let agents = make_agents_with_vel(&[
            (0.0, 0.0, 1.0, 0.0),
            (1.0, 0.0, 1.0, 0.0),
            (2.0, 0.0, 1.0, 0.0),
        ]);
        assert!(approx_eq(swarm_order(&agents), 1.0));
    }

    #[test]
    fn test_order_opposite() {
        let agents = make_agents_with_vel(&[(0.0, 0.0, 1.0, 0.0), (1.0, 0.0, -1.0, 0.0)]);
        assert!(approx_eq(swarm_order(&agents), 0.0));
    }

    #[test]
    fn test_order_empty() {
        assert!(approx_eq(swarm_order(&[]), 0.0));
    }

    #[test]
    fn test_order_zero_velocity() {
        let agents = make_agents(&[(0.0, 0.0), (1.0, 0.0)]);
        assert!(approx_eq(swarm_order(&agents), 0.0));
    }

    #[test]
    fn test_min_distance() {
        let agents = make_agents(&[(0.0, 0.0), (3.0, 4.0), (10.0, 0.0)]);
        assert!(approx_eq(swarm_min_distance(&agents), 5.0));
    }

    #[test]
    fn test_min_distance_single() {
        let agents = make_agents(&[(0.0, 0.0)]);
        assert!(approx_eq(swarm_min_distance(&agents), 0.0));
    }

    #[test]
    fn test_min_distance_empty() {
        assert!(approx_eq(swarm_min_distance(&[]), 0.0));
    }

    #[test]
    fn test_diameter() {
        let agents = make_agents(&[(0.0, 0.0), (3.0, 0.0), (6.0, 0.0)]);
        assert!(approx_eq(swarm_diameter(&agents), 6.0));
    }

    #[test]
    fn test_diameter_single() {
        let agents = make_agents(&[(5.0, 5.0)]);
        assert!(approx_eq(swarm_diameter(&agents), 0.0));
    }

    #[test]
    fn test_collision_count() {
        let agents = make_agents(&[(0.0, 0.0), (0.5, 0.0), (10.0, 0.0)]);
        assert_eq!(swarm_collision_count(&agents, 1.0), 1);
    }

    #[test]
    fn test_collision_count_none() {
        let agents = make_agents(&[(0.0, 0.0), (10.0, 0.0)]);
        assert_eq!(swarm_collision_count(&agents, 1.0), 0);
    }

    #[test]
    fn test_connectivity_full() {
        let agents = make_agents(&[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]);
        assert!(approx_eq(swarm_connectivity(&agents, 10.0), 1.0));
    }

    #[test]
    fn test_connectivity_partial() {
        let agents = make_agents(&[(0.0, 0.0), (1.0, 0.0), (100.0, 0.0)]);
        // Only pair (0,1) within range 2.0. Total pairs=3.
        assert!(approx_eq(swarm_connectivity(&agents, 2.0), 1.0 / 3.0));
    }

    #[test]
    fn test_connectivity_single() {
        let agents = make_agents(&[(0.0, 0.0)]);
        assert!(approx_eq(swarm_connectivity(&agents, 1.0), 1.0));
    }

    #[test]
    fn test_connectivity_empty() {
        assert!(approx_eq(swarm_connectivity(&[], 1.0), 1.0));
    }

    // --- Integration tests ---

    #[test]
    fn test_boids_with_obstacles() {
        let mut agents = make_agents_with_vel(&[
            (0.0, 0.0, 2.0, 0.0),
            (0.0, 1.0, 2.0, 0.0),
            (0.0, -1.0, 2.0, 0.0),
        ]);
        let obstacles = vec![Obstacle::new(Vec2::new(5.0, 0.0), 2.0)];
        let params = BoidParams::default();

        for _ in 0..100 {
            let mut steers: Vec<Vec2> = (0..agents.len())
                .map(|i| boids_steer(&agents, i, &params))
                .collect();
            for (i, s) in steers.iter_mut().enumerate() {
                *s = *s + obstacle_avoidance(&agents[i], &obstacles, 4.0, 5.0);
            }
            for (a, s) in agents.iter_mut().zip(steers.iter()) {
                a.velocity = (a.velocity + *s * 0.05).clamped(params.max_speed);
                a.position = a.position + a.velocity * 0.05;
            }
        }
        // No agent should be inside the obstacle
        for a in &agents {
            assert!(a.position.distance_to(obstacles[0].center) > obstacles[0].radius - 0.5);
        }
    }

    #[test]
    fn test_consensus_with_star_topology() {
        let topo = Topology::star(5);
        let mut values = vec![10.0, 0.0, 0.0, 0.0, 0.0];
        for _ in 0..2000 {
            consensus_step(&mut values, &topo, 0.3);
        }
        // All values should converge to the same value
        let converged = values[0];
        for &v in &values[1..] {
            assert!((v - converged).abs() < 0.1);
        }
    }

    #[test]
    fn test_formation_and_metrics() {
        let f = Formation::ring(6, 5.0);
        let mut agents: Vec<Agent> = (0..6)
            .map(|i| Agent::new(i, Vec2::zero(), Vec2::zero()))
            .collect();
        let center = Vec2::new(10.0, 10.0);

        for _ in 0..100 {
            let steers = formation_steer(&agents, &f, center, 0.5);
            for (a, s) in agents.iter_mut().zip(steers.iter()) {
                a.velocity = *s;
                a.position = a.position + *s * 0.1;
            }
        }

        let c = swarm_centroid(&agents);
        assert!((c.x - 10.0).abs() < 1.0);
        assert!((c.y - 10.0).abs() < 1.0);
    }

    #[test]
    fn test_large_swarm_metrics() {
        let agents: Vec<Agent> = (0..100)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let x = (i % 10) as f64;
                #[allow(clippy::cast_precision_loss)]
                let y = (i / 10) as f64;
                Agent::new(i, Vec2::new(x, y), Vec2::new(1.0, 0.0))
            })
            .collect();
        let c = swarm_centroid(&agents);
        assert!(approx_eq(c.x, 4.5));
        assert!(approx_eq(c.y, 4.5));
        assert!(approx_eq(swarm_order(&agents), 1.0));
        assert!(swarm_spread(&agents) > 0.0);
        assert!(swarm_min_distance(&agents) <= 1.0 + EPS);
    }

    #[test]
    fn test_boid_params_default() {
        let p = BoidParams::default();
        assert!(p.max_speed > 0.0);
        assert!(p.max_force > 0.0);
    }

    #[test]
    fn test_obstacle_new() {
        let o = Obstacle::new(Vec2::new(1.0, 2.0), 3.0);
        assert!(approx_eq(o.center.x, 1.0));
        assert!(approx_eq(o.radius, 3.0));
    }

    #[test]
    fn test_task_new() {
        let t = Task::new(7, Vec2::new(1.0, 2.0), 5.0);
        assert_eq!(t.id, 7);
        assert!(approx_eq(t.priority, 5.0));
    }

    #[test]
    fn test_topology_is_connected_false() {
        let t = Topology::ring(5);
        assert!(!t.is_connected(0, 2));
    }

    #[test]
    fn test_formation_line_spacing() {
        let f = Formation::line(3, 4.0);
        // Offsets: -4, 0, 4
        assert!(approx_eq(f.offsets[0].x, -4.0));
        assert!(approx_eq(f.offsets[1].x, 0.0));
        assert!(approx_eq(f.offsets[2].x, 4.0));
    }

    #[test]
    fn test_vec2_equality() {
        let a = Vec2::new(1.0, 2.0);
        let b = Vec2::new(1.0, 2.0);
        assert_eq!(a, b);
    }

    #[test]
    fn test_agent_clone() {
        let a = Agent::new(0, Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
        let b = a.clone();
        assert_eq!(a.id, b.id);
        assert_eq!(a.position, b.position);
    }

    #[test]
    fn test_vec2_debug() {
        let v = Vec2::new(1.0, 2.0);
        let s = format!("{v:?}");
        assert!(s.contains("1.0"));
    }

    #[test]
    fn test_leader_follower_stationary_leader() {
        let leader = Agent::new(0, Vec2::new(5.0, 0.0), Vec2::zero());
        let follower = Agent::new(1, Vec2::new(0.0, 0.0), Vec2::zero());
        let steer = leader_follower_steer(&follower, &leader, 2.0, 1.0);
        // Leader stationary -> fallback direction (1,0), target = (3,0)
        assert!(steer.x > 0.0);
    }

    #[test]
    fn test_consensus_preserves_sum() {
        let topo = Topology::fully_connected(4);
        let mut values = vec![1.0, 2.0, 3.0, 4.0];
        let initial_sum: f64 = values.iter().sum();
        consensus_step(&mut values, &topo, 0.5);
        let new_sum: f64 = values.iter().sum();
        assert!((initial_sum - new_sum).abs() < 1e-10);
    }

    #[test]
    fn test_greedy_allocation_three_agents() {
        let agents = make_agents(&[(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)]);
        let tasks = vec![
            Task::new(0, Vec2::new(0.5, 0.0), 1.0),
            Task::new(1, Vec2::new(5.5, 0.0), 1.0),
            Task::new(2, Vec2::new(10.5, 0.0), 1.0),
        ];
        let alloc = allocate_greedy(&agents, &tasks);
        assert_eq!(alloc.len(), 3);
        assert_eq!(alloc[&0], 0);
        assert_eq!(alloc[&1], 1);
        assert_eq!(alloc[&2], 2);
    }

    #[test]
    fn test_path_blocked_tangent() {
        // Path goes right past the edge of obstacle
        let obstacles = vec![Obstacle::new(Vec2::new(5.0, 1.0), 1.0)];
        // Straight line along x-axis at y=0, closest approach = 1.0 = radius (tangent)
        assert!(path_blocked(
            Vec2::new(0.0, 0.0),
            Vec2::new(10.0, 0.0),
            &obstacles
        ));
    }

    #[test]
    fn test_swarm_diameter_empty() {
        assert!(approx_eq(swarm_diameter(&[]), 0.0));
    }

    #[test]
    fn test_collision_count_all_close() {
        let agents = make_agents(&[(0.0, 0.0), (0.1, 0.0), (0.2, 0.0)]);
        // All 3 pairs within 1.0
        assert_eq!(swarm_collision_count(&agents, 1.0), 3);
    }
}
