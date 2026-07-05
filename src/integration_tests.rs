//! Cross-module integration tests.

#![allow(
    clippy::doc_markdown,
    clippy::assertions_on_constants,
    clippy::suboptimal_flops,
    clippy::unreadable_literal,
    clippy::float_cmp,
    clippy::similar_names,
    clippy::needless_collect,
    clippy::case_sensitive_file_extension_comparisons,
    clippy::redundant_clone,
    clippy::needless_range_loop,
    clippy::cast_lossless,
    clippy::manual_range_contains,
    clippy::should_panic_without_expect,
    clippy::assign_op_pattern
)]

use core::f64;

use crate::agent::*;
use crate::boids::*;
use crate::formation::*;
use crate::leader::*;
use crate::metrics::*;
use crate::obstacle::*;
use crate::task::*;
use crate::topology::*;
use crate::vec2::*;

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

// -- Vec2 --

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

// -- Agent --

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

// -- Boids separation --

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
    assert!(s.x < 0.0);
}

#[test]
fn test_boids_separation_symmetry() {
    let agents = make_agents(&[(0.0, 0.0), (1.0, 0.0)]);
    let s0 = boids_separation(&agents, 0, 5.0);
    let s1 = boids_separation(&agents, 1, 5.0);
    assert!(approx_eq(s0.x + s1.x, 0.0));
}

// -- Boids alignment --

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
    assert!(a.x < 0.0);
}

#[test]
fn test_boids_alignment_no_neighbors() {
    let agents = make_agents_with_vel(&[(0.0, 0.0, 1.0, 0.0), (100.0, 0.0, -1.0, 0.0)]);
    let a = boids_alignment(&agents, 0, 5.0);
    assert!(approx_eq(a.length(), 0.0));
}

// -- Boids cohesion --

#[test]
fn test_boids_cohesion_toward_group() {
    let agents = make_agents(&[(0.0, 0.0), (2.0, 0.0), (2.0, 2.0)]);
    let c = boids_cohesion(&agents, 0, 10.0);
    assert!(c.x > 0.0);
    assert!(c.y > 0.0);
}

#[test]
fn test_boids_cohesion_no_neighbors() {
    let agents = make_agents(&[(0.0, 0.0), (100.0, 100.0)]);
    let c = boids_cohesion(&agents, 0, 5.0);
    assert!(approx_eq(c.length(), 0.0));
}

// -- Boids steer --

#[test]
fn test_boids_steer_returns_clamped() {
    let agents = make_agents(&[(0.0, 0.0), (0.5, 0.0), (0.0, 0.5)]);
    let params = BoidParams::default();
    let s = boids_steer(&agents, 0, &params);
    assert!(s.length() <= params.max_force + EPS);
}

// -- Boids step --

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
    for a in &agents {
        assert!(a.position.length() < 1000.0);
    }
}

// -- Formation --

#[test]
fn test_formation_line() {
    let f = Formation::line(5, 2.0);
    assert_eq!(f.slot_count(), 5);
    let sum: f64 = f.offsets.iter().map(|o| o.x).sum();
    assert!(approx_eq(sum, 0.0));
}

#[test]
fn test_formation_ring() {
    let f = Formation::ring(4, 10.0);
    assert_eq!(f.slot_count(), 4);
    for o in &f.offsets {
        assert!(approx_eq(o.length(), 10.0));
    }
}

#[test]
fn test_formation_v_shape() {
    let f = Formation::v_shape(5, 2.0, f64::consts::FRAC_PI_4);
    assert_eq!(f.slot_count(), 5);
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

// -- Topology --

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
    assert_eq!(t.neighbors[0].len(), 4);
    for i in 1..5 {
        assert_eq!(t.neighbors[i].len(), 1);
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

// -- Consensus --

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

// -- Task allocation --

#[test]
fn test_greedy_allocation() {
    let agents = make_agents(&[(0.0, 0.0), (10.0, 0.0)]);
    let tasks = vec![
        Task::new(0, Vec2::new(1.0, 0.0), 1.0),
        Task::new(1, Vec2::new(9.0, 0.0), 1.0),
    ];
    let alloc = allocate_greedy(&agents, &tasks);
    assert_eq!(alloc[&0], 0);
    assert_eq!(alloc[&1], 1);
}

#[test]
fn test_greedy_allocation_priority() {
    let agents = make_agents(&[(5.0, 0.0)]);
    let tasks = vec![
        Task::new(0, Vec2::new(0.0, 0.0), 1.0),
        Task::new(1, Vec2::new(10.0, 0.0), 10.0),
    ];
    let alloc = allocate_greedy(&agents, &tasks);
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

// -- Obstacle avoidance --

#[test]
fn test_obstacle_avoidance_repels() {
    let agent = Agent::new(0, Vec2::new(2.0, 0.0), Vec2::new(0.0, 0.0));
    let obstacles = vec![Obstacle::new(Vec2::new(0.0, 0.0), 1.0)];
    let force = obstacle_avoidance(&agent, &obstacles, 5.0, 1.0);
    assert!(force.x > 0.0);
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

// -- Leader-follower --

#[test]
fn test_leader_follower_steer_behind() {
    let leader = Agent::new(0, Vec2::new(10.0, 0.0), Vec2::new(1.0, 0.0));
    let follower = Agent::new(1, Vec2::new(0.0, 0.0), Vec2::zero());
    let steer = leader_follower_steer(&follower, &leader, 2.0, 1.0);
    assert!(steer.x > 0.0);
}

#[test]
fn test_leader_follower_steer_at_target() {
    let leader = Agent::new(0, Vec2::new(5.0, 0.0), Vec2::new(1.0, 0.0));
    let follower = Agent::new(1, Vec2::new(3.0, 0.0), Vec2::zero());
    let steer = leader_follower_steer(&follower, &leader, 2.0, 1.0);
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
    assert!(agents[0].position.x > 5.0);
    assert!(agents[1].position.x < agents[0].position.x);
    assert!(agents[2].position.x < agents[1].position.x);
}

#[test]
fn test_leader_follower_step_empty() {
    let mut agents: Vec<Agent> = vec![];
    leader_follower_step(&mut agents, Vec2::new(1.0, 0.0), 2.0, 1.0, 0.1);
    assert!(agents.is_empty());
}

// -- Swarm metrics --

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
    assert!(approx_eq(s, 2.5));
}

#[test]
fn test_avg_speed_empty() {
    assert!(approx_eq(swarm_avg_speed(&[]), 0.0));
}

#[test]
fn test_spread() {
    let agents = make_agents(&[(0.0, 0.0), (2.0, 0.0)]);
    let s = swarm_spread(&agents);
    assert!(approx_eq(s, 1.0));
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

// -- Integration --

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
    let obstacles = vec![Obstacle::new(Vec2::new(5.0, 1.0), 1.0)];
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
    assert_eq!(swarm_collision_count(&agents, 1.0), 3);
}
