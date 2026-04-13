# CS780 - Capstone project
## OBELIX - The Warehouse Robot
## Task Description

This project addresses the OBELIX warehouse robot task, where a reinforcement learning agent must learn to **locate, attach to, and push a grey box to the arena boundary** using limited sensor information.

### Environment

The robot operates with a fixed set of **5 discrete actions**:
- Rotate right (45°)
- Rotate right (22°)
- Move forward
- Rotate left (22°)
- Rotate left (45°)

It receives an **18-bit observation vector** consisting of:
- 16 sonar sensor bits (near and far range)
- 1 infrared (IR) sensor bit
- 1 stuck indicator (wall/boundary)

The environment is **partially observable (POMDP)**, as the agent only has access to local sensor readings and not the full state.

### Objective

The agent must:
1. Find the grey box  
2. Attach to it (automatic on contact)  
3. Push it to the arena boundary  
4. Detach once the box reaches the boundary  

Episodes terminate upon successful completion or when a maximum step limit is reached. Performance is evaluated using cumulative reward.

### Challenges

- Partial observability and state aliasing  
- Sparse and delayed rewards  
- Exploration vs. exploitation trade-off  
- Handling wall interactions and stuck conditions  

### Difficulty Levels

- **Level 1 – Static Box:** Box remains stationary  
- **Level 2 – Blinking Box:** Box appears/disappears randomly  
- **Level 3 – Moving + Blinking Box:** Box moves and blinks until attached  

Additionally, a wall with a narrow opening may be present, requiring careful navigation.

### Note

The task can be conceptually divided into *Find → Push → Unwedge*, but these states are not explicitly provided to the agent and must be inferred from observations.
