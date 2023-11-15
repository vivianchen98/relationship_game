# congestion game: two-lane roads of the same direction
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
    phi = [ [1 0 0; 0 1 0; 0 0 1],  # each car's selfish utilities
            [0 1 1; 1 0 1; 1 1 0]]  # regular car to regular car
    
    (; u = u, V = V, phi = phi, λ = 0.7)
end

function rps()
    N = 2; dim=[3,3];

    # create utilities u for each player
    u = [ 0 -1  1;
          1  0 -1;
         -1  1  0 ]
    u = [u , -u]
    
    return N, u
end

function pd()
    N = 2; dim=[2,2];

    # create utilities u for each player
    u = [1 3;
         0 2]
    u = [u , u']

    return N, u
end

function bee_queen()
    N = 4; dim=[2,2,2,2];

    # road delay functions
    a(load) = load
    b(load) = 1.5*load
    func_list = [a,b]

    # create utilities u for each player
    u = [zeros(dim...) for i = 1:N]
    for i=1:2, j=1:2, k=1:2, l=1:2
        for n=1:N
            player = [i,j,k,l][n]
            load = count(x->(x==player), [i,j,k,l])
            u[n][i,j,k,l] = func_list[player](load)
        end
    end

    # social cost as the weighted sum of utilities
    w = [2, 1, 1, 1]
    V = sum(w[i] * u[i] for i in 1:N)

    # relationship structure imposed by reality
    phi = [ [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1],   # selifsh utility: identity matrix
            [0 0 0 0; 0 0 1 1; 0 1 0 1; 0 1 1 0],   # regular car to regular car
            [0 0 0 0; 1 0 0 0; 1 0 0 0; 1 0 0 0],   # regular car to ambulance
            [0 1 1 1; 0 0 0 0; 0 0 0 0; 0 0 0 0]    # ambulance to regular car
          ]

    (; u = u, V = V, phi = phi)
end

function bee_queen(N)
    N = N; dim=repeat([2], N);

    # road delay functions
    a(load) = load
    b(load) = 2*load
    func_list = [a,b]

    # create utilities u for each player
    u = [zeros(dim...) for i = 1:N]
    for i=1:2, j=1:2, k=1:2, l=1:2
        for n=1:N
            player = [i,j,k,l][n]
            load = count(x->(x==player), [i,j,k,l])
            u[n][i,j,k,l] = func_list[player](load)
        end
    end

    # social cost as the weighted sum of utilities
    w = [5, 1, 1, 1]
    V = sum(u[i] * w[i] for i in 1:N)

    # relationship structure imposed by reality
    # phi = [[0 1 1; 1 0 1; 1 1 0],
    #        [1 0 0; 0 0 0; 0 0 0],
    #        [0 0 0; 0 1 0; 0 0 0],
    #        [0 0 0; 0 0 0; 0 0 1]]

    phi = [[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1],
           [0 1 1 1; 1 0 1 1; 1 1 0 1; 1 1 1 0]]

    (; u = u, V = V, phi = phi)
end

# "undivided road": traffic flows in opposite directions and is typically separated by only lane markings, without any physical barrier or median.
function undivided_congestion()
    N = 6; dim=[2,2,2,2,2,2];

    # road delay functions
    # a(load) = load
    # b(load) = 2*load
    # func_list = [a,b]

    # create utilities u for each player
    u = [zeros(dim...) for i = 1:N]
    for i=1:2, j=1:2, k=1:2, l=1:2, m=1:2, n=1:2
        for o=1:N
            player = [i,j,k,l,m,n][o]
            load = count(x->(x==player), [i,j,k,l,m,n])
            u[o][i,j,k,l,m,n] = func_list[player](load)
        end
    end

    # social cost as the sum of utilities
    V = sum(u[i] for i in 1:N)

    # relationship structure imposed by reality
    phi = [[1 0 0 0 0 0; 0 1 0 0 0 0; 0 0 1 0 0 0; 0 0 0 1 0 0; 0 0 0 0 1 0; 0 0 0 0 0 1],
           [0 1 1 1 1 1; 1 0 1 1 1 1; 1 1 0 1 1 1; 1 1 1 0 1 1; 1 1 1 1 0 1; 1 1 1 1 1 0]]

    (; u = u, V = V, phi = phi)
end