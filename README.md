# relationship_game

## Description
To design weight vector on a relationship game (RG) so that is solution is socially desirable.

## Getting Started
### Installing

### Running

#### Global Method: order-and-design
* Given a RG, compute a sparse w using a LP, solved by Gurobi
```
julia order_and_design.jl
```



#### Gradient Method
* New game solver: Entropy-Regularized Nash Solver in
```
game_solvers/entropy_nash_solver.jl
```
* You can visualize the effect of entropy nash solver by enter julia, then do
```
julia> include("visualize_entropy_nash.jl")
```

### Acknowledgments
