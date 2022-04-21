


# order and design
function order_and_design(A1, A2, u, V, phi, k)
    # order
    unvisited = vec(collect(Iterators.product(A1, A2)))

    while !isempty(unvisited)
        a = convert(Tuple, argmax(V))
        (a1, a2) = a

       # call design
        found, w, eps, V = design(A1, A2, u, a1, a2, , phi, k)

        if found
            return w
        else
            deleteat!(unvisited, findall(x->x==a, unvisited))
        end
    end

    return false
end

A1 = 1:2
A2 = 1:2
# u = [[3 0; 5 1], [3 5; 0 1]]
u = [[2 4; 1.5 3], [2 1.5; 4 3]]
V = [3 2; 2 1]
phi = [[0 1; 1 0], [0 1; 0 0]]
k = 3

w = order_and_design(A1, A2, u, V, phi, k)
print(w)