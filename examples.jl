# congestion game
function congestion()
    # 3 players on a two-path road
    N = 3; dim=[2,2,2];

    # road delay functions
    a(load) = load
    b(load) = 2*load
    func_list = [a,b]

    # create utilities u for each player
    u = [zeros(dim...) for i = 1:N]
    for i=1:2, j=1:2, k=1:2
        for n=1:N
            player = [i,j,k][n]
            load = count(x->(x==player), [i,j,k])
            u[n][i,j,k] = func_list[player](load)
        end
    end

    # social cost as the sum of utilities
    V = sum(u[i] for i in 1:N)

    # relationship structure imposed by reality
    # phi = [[0 1 1; 1 0 1; 1 1 0],
    #        [1 0 0; 0 0 0; 0 0 0],
    #        [0 0 0; 0 1 0; 0 0 0],
    #        [0 0 0; 0 0 0; 0 0 1]]
    
    # phi = [[1 0 0; 0 1 0; 0 0 1],
    #        [0 1 1; 1 0 1; 1 1 0]]
    
    phi = [[0 1 1; 1 0 1; 1 1 0]]

    (; u = u, V = V, phi = phi)
end