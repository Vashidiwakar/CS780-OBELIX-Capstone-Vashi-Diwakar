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

The environment is **partially observable (POMDP)**.

### Objective

1. Find the grey box  
2. Attach to it  
3. Push it to the arena boundary  
4. Detach  

### Challenges

- Partial observability  
- Sparse rewards  
- Exploration vs exploitation  

### Difficulty Levels

- Static box  
- Blinking box  
- Moving + blinking box  

### Note

Find → Push → Unwedge