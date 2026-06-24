# OOD Findings: 

This document covers the OOD results from the Trakr Baseline in the same OOD tasks as the Unitree Go2 Rough AsymPPO checkpoint.

## Reading Rule

For OOD results, the metrics considered are identical to the metrics used to evaluate the Rough AsymPPO checkpoint:

- velocity_error
- `timeout_fraction_of_terminals`
- `base_contact_fraction_of_terminals`
- `bad_orientation_fraction_of_terminals`

Higher timeout fraction is better.
Higher base-contact fraction and bad-orientation fraction is worse.

## Geometry OOD Summary

### Stairs Down

- Trakr AsymPPO:
  - `episodes = 98`
  - `vel_err = 0.1908`
  - `timeout_frac = 0.9479`
  - `base_contact_frac = 0.0000`
  - `bad_orientation_frac = 0.0521`

Visual Comparison:

- Unitree Rough AsymPPO checkpoint can only traverse low step height stairs.
- Even for low step height, the success rate is around 75%
- For greater step heights, the Rough AsymPPO checkpoint cannot handle the high momentum and topples before reaching the bottom
- Trakr AsymPPO checkpoint can handle step heights upto 0.18m with 95% success rate.
- Trakr checkpoint can traverse down the stairs with a stable gait and low momentum.

### Random Rough Level 9


### Boxes

- Trakr AsymPPO:
  - `episodes = 9106`
  - `vel_err = 0.2704`
  - `timeout_frac = 0.5094`
  - `base_contact_frac = 0.0188`
  - `bad_orientation_frac = 0.4716`

Visual Comparison:

- Unitree Rough AsymPPO checkpoint can only traverse through very small height boxes
- It cannot lift its legs to traverse over small step height obstacles.
- Trakr AsymPPO checkpoint can lift its legs to traverse through obstacles of step height <= 0.18m
- It cannot lift its legs to overcome obstacles of height > 0.18m
- Both checkpoints cannot anticipate gaps in the terain.
- Unitree Rough AsymPPO checkpoint can recover from situations where it has fallen on its base and stand up straight, but the Trakr AsymPPO checkpoint cannot.

### Stairs Up


## Dynamics OOD Summary

### Ultra-High Friction


### Very Heavy


### Ultra-Low Friction


### Very Weak Motors


## Push Recovery OOD Summary

### Yaw Push Medium


### Forward Push Medium

### Lateral Push Medium

### Lateral Push Repeated

### Push Recovery Takeaway

## Switch OOD Summary

### Switch To Ultra-Low Friction

### Switch To Low-Friction + Heavy

### Switch To Very Heavy

### Switch To Very Weak Motors


