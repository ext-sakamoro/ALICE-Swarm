//! Leader-follower dynamics.

use crate::agent::Agent;
use crate::vec2::Vec2;

/// Compute follower steering to track a leader agent.
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
    agents[0].velocity = leader_velocity;
    agents[0].position += leader_velocity * dt;

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
