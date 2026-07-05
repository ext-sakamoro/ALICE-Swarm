//! Boids algorithm.

use crate::agent::Agent;
use crate::vec2::Vec2;

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
