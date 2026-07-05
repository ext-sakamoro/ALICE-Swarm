//! `Obstacle` + obstacle avoidance.

use crate::agent::Agent;
use crate::vec2::Vec2;

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
