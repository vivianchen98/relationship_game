# script to generate N-player game matrix of traffic example

function generate_traffic(N=3, dim=[2,2,2])
    @assert N<=3 "Only support at most 3 players!"
    # @assert maximum(dim) <= 31 "Only support at most 31 actions!"
    @assert N==length(dim) "dimension for actions must match the number of players!"

    a(load)= 4 / load
    b(load) = 1.5 * load
    c(load) = 2 * load
    d(load) = 2.5 * load
    e(load) = 3 * load
    f(load) = 3.5 * load

    func_list = [[b,c,d,e,f] for i in 1:(maximum(dim)÷5+1)]; func_list = collect(Iterators.flatten(func_list))
    func_list = pushfirst!(func_list, a)

    cost_tensors = [zeros(dim...) for i = 1:N]
    
    if N == 2
        for i=1:dim[1], j=1:dim[2]
            for n=1:N
                player = [i,j][n]
                load = count(x->(x==player), [i,j])
                cost_tensors[n][i,j] = func_list[player](load)
            end
        end

    elseif N == 3
        for i=1:dim[1], j=1:dim[2], k=1:dim[3]
            for n=1:N
                player = [i,j,k][n]
                load = count(x->(x==player), [i,j,k])
                # cost_tensors[n][i,j,k] = (player==1) ? a(load) : b(load)
                cost_tensors[n][i,j,k] = func_list[player](load)
            end
        end
    end

    # cost_tensors = [cost_tensors[1];;;cost_tensors[2];;;cost_tensors[3]]
    return cost_tensors
end

