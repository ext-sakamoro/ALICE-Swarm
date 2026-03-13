**English** | [日本語](README_JP.md)

# ALICE-Swarm

Multi-agent swarm control library for the ALICE ecosystem. Provides Boids algorithm, formation control, consensus protocols, task allocation, obstacle avoidance, and swarm metrics -- all in pure Rust.

## Features

- **Boids Algorithm** -- Separation, alignment, cohesion with configurable radii and weights
- **Formation Control** -- Line, ring, V-shape, and grid formations with formation-steering
- **Consensus Protocols** -- Scalar and vector consensus over arbitrary communication topologies
- **Communication Topology** -- Fully connected, ring, star, k-nearest-neighbor graphs
- **Task Allocation** -- Greedy and auction-based assignment of tasks to agents
- **Obstacle Avoidance** -- Repulsive potential field, path blocking detection
- **Leader-Follower** -- Leader tracking with configurable follow distance and gain
- **Swarm Metrics** -- Centroid, average speed, spread, order parameter, min distance, diameter, collision count, connectivity

## Architecture

```
Agent (position, velocity, max_speed, max_force)
    |
    +-- Boids (separation + alignment + cohesion)
    +-- Formation (line / ring / V / grid)
    +-- LeaderFollower (leader tracking)
    |
    v
Topology (fully connected / ring / star / k-NN)
    |
    v
Consensus (scalar / vector convergence)
    |
    v
Task Allocation (greedy / auction)
    |
    v
Obstacle Avoidance --> Swarm Metrics
```

## License

MIT OR Apache-2.0
