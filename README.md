# ethical_game

## Description
Compute the strategy profile with the maximum collective utility (ethical) of the a game based on a multi-player ethical game framework.

## Getting Started
### Installing
dependencies: argparse

### Running experiment

* Partial info (player only updates the selected entry in game matrix) needs epsilon-nash to boost exploration
```
python3 main_master.py --iter 100 --gamma 0.1 --epsilon 0.1 --nash_type epsilon_nash --player_info partial
```

* Complete info (player can update all entries in game matrix) does not need much exploration, can just use nash
```
python3 main_master.py --iter 100 --gamma 0.1 -nash_type nash --player_info complete
```


command line arguments:
* --iter(int): the number of simulation iterations
* --gamma(float): the selfish/altruistic factor
* --nash_type(str)={'nash', 'epsilon_nash'}: Nash solver or epsilon-greedy Nash solver
* --epsilon(float): epsilon parameter if epsilon_nash
* --scores_type(str)-{'original', 'current'}: if the argmax step is based on original game matrix or the matrix at the current iteration
* --player_info(str)={'complete', 'partial'}: if in iteration the game matrix is updated in all entries (assuming players have complete information about the game), or is updated only in the strategy profile they commited (assuming players only have partial information)

### Acknowledgments

* [ipd](https://github.com/cristal-smac/ipd): game class where the Nash solver is defined in game.py



To do:
- [x] set up IPD exp
- [x] prelim implement algo
- [x] write game class in a way that is convenient for the algorithm
- [ ] look into Nash solver
- [ ] add trembling-hand?
- [x] add epsilon_Nash
- [x] update all entries at once
- [x] constrain alpha so that the sum of scores non-decreasing (unitary, now identity matrix)
