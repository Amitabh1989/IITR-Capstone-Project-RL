# OpenAI Gym : Frozenlake_reinforcement_learning (Q learning and Deep Q Learning)

# Code Author : Amitabh Suman
README from https://www.gymlibrary.dev/environments/toy_text/frozen_lake/

<img src="https://user-images.githubusercontent.com/12171805/229715773-d1411961-5b93-4c65-9beb-de79e8513705.png" width="400" height="250"> and <img src="https://user-images.githubusercontent.com/12171805/229716086-c7f1ca1b-330c-462f-a03b-462561abb1fc.png" width="250" height="250">


This environment is part of the Toy Text environments. Please read that page first for general information.

## **Action Space**

Discrete(4)

## **Observation Space**

Discrete(16)

Import

gym.make("FrozenLake-v1")

Frozen lake involves crossing a frozen lake from Start(S) to Goal(G) without falling into any Holes(H) by walking over the Frozen(F) lake. The agent may not always move in the intended direction due to the slippery nature of the frozen lake.

## **Action Space**
The agent takes a 1-element vector for actions. The action space is (dir), where dir decides direction to move in which can be:

0: LEFT

1: DOWN

2: RIGHT

3: UP

## **Observation Space**
The observation is a value representing the agent’s current position as current_row * nrows + current_col (where both the row and col start at 0). For example, the goal position in the 4x4 map can be calculated as follows: 3 * 4 + 3 = 15. The number of possible observations is dependent on the size of the map. For example, the 4x4 map has 16 possible observations.

Rewards
Reward schedule:

Reach goal(G): +1

Reach hole(H): 0

Reach frozen(F): 0

## **Arguments**
gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
desc: Used to specify custom map for frozen lake. For example,

desc=["SFFF", "FHFH", "FFFH", "HFFG"].

A random generated map can be specified by calling the function `generate_random_map`. For example,

```
from gym.envs.toy_text.frozen_lake import generate_random_map

gym.make('FrozenLake-v1', desc=generate_random_map(size=8))
```
map_name: ID to use any of the preloaded maps.

"4x4":[
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG"
    ]

"8x8": [
    "SFFFFFFF",
    "FFFFFFFF",
    "FFFHFFFF",
    "FFFFFHFF",
    "FFFHFFFF",
    "FHHFFFHF",
    "FHFFHFHF",
    "FFFHFFFG",
]
is_slippery: True/False. If True will move in intended direction with probability of 1/3 else will move in either perpendicular direction with equal probability of 1/3 in both directions.

For example, if action is left and is_slippery is True, then:
- P(move left)=1/3
- P(move up)=1/3
- P(move down)=1/3
