# script to generate N-player game matrix of traffic example

function generate_traffic(N=3, dim=[2,2,2])
    @assert N<=20 "Only support at most 20 players!"
    # @assert maximum(dim) <= 31 "Only support at most 31 actions!"
    @assert N==length(dim) "dimension for actions must match the number of players!"

    M = dim[1]
    a(load)= (5/3 * N)/ load
    # b(load) = (a_val/N + 0.5) * load
    b(load) = 1.5 * load
    c(load) = 2  * load
    d(load) = 4 * load
    e(load) = 4.5  * load
    f(load) = 5 * load

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
    elseif N == 4
        for i=1:dim[1], j=1:dim[2], k=1:dim[3], l=1:dim[4]
            for n=1:N
                player = [i,j,k,l][n]
                load = count(x->(x==player), [i,j,k,l])
                cost_tensors[n][i,j,k,l] = func_list[player](load)
            end
        end
    elseif N == 5
        for i=1:dim[1], j=1:dim[2], k=1:dim[3], l=1:dim[4], m=1:dim[5]
            for n=1:N
                player = [i,j,k,l,m][n]
                load = count(x->(x==player), [i,j,k,l,m])
                cost_tensors[n][i,j,k,l,m] = func_list[player](load)
            end
        end
    elseif N == 6
        for i=1:dim[1], j=1:dim[2], k=1:dim[3], l=1:dim[4], m=1:dim[5], o=1:dim[6]
            for n=1:N
                player = [i,j,k,l,m, o][n]
                load = count(x->(x==player), [i,j,k,l,m,o])
                cost_tensors[n][i,j,k,l,m,o] = func_list[player](load)
            end
        end      
    elseif N == 7
        for i=1:dim[1], j=1:dim[2], k=1:dim[3], l=1:dim[4], m=1:dim[5], o=1:dim[6], p=1:dim[7]
            for n=1:N
                player = [i,j,k,l,m,o,p][n]
                load = count(x->(x==player), [i,j,k,l,m,o,p])
                cost_tensors[n][i,j,k,l,m,o,p] = func_list[player](load)
            end
        end
    elseif N == 8
        for i=1:dim[1], j=1:dim[2], k=1:dim[3], l=1:dim[4], m=1:dim[5], o=1:dim[6], p=1:dim[7], q=1:dim[8]
            for n=1:N
                player = [i,j,k,l,m,o,p,q][n]
                load = count(x->(x==player), [i,j,k,l,m,o,p,q])
                cost_tensors[n][i,j,k,l,m,o,p,q] = func_list[player](load)
            end
        end      
    elseif N == 9
        for i=1:dim[1], j=1:dim[2], k=1:dim[3], l=1:dim[4], m=1:dim[5], o=1:dim[6], p=1:dim[7], q=1:dim[8], r=1:dim[9]
            for n=1:N
                player = [i,j,k,l,m,o,p,q,r][n]
                load = count(x->(x==player), [i,j,k,l,m,o,p,q,r])
                cost_tensors[n][i,j,k,l,m,o,p,q,r] = func_list[player](load)
            end
        end  
    elseif N == 10
        for i=1:dim[1], j=1:dim[2], k=1:dim[3], l=1:dim[4], m=1:dim[5], o=1:dim[6], p=1:dim[7], q=1:dim[8], r=1:dim[9], s=1:dim[10]
            for n=1:N
                player = [i,j,k,l,m,o,p,q,r,s][n]
                load = count(x->(x==player), [i,j,k,l,m,o,p,q,r,s])
                cost_tensors[n][i,j,k,l,m,o,p,q,r,s] = func_list[player](load)
            end
        end  
    elseif N == 15
        for i=1:dim[1], j=1:dim[2], k=1:dim[3], l=1:dim[4], m=1:dim[5], o=1:dim[6], p=1:dim[7], q=1:dim[8], r=1:dim[9], s=1:dim[10], t=1:dim[11], u=1:dim[12], v=1:dim[13], w=1:dim[14], x=1:dim[15]
            for n=1:N
                player = [i,j,k,l,m,o,p,q,r,s,t,u,v,w,x][n]
                load = count(x->(x==player), [i,j,k,l,m,o,p,q,r,s,t,u,v,w,x])
                cost_tensors[n][i,j,k,l,m,o,p,q,r,s,t,u,v,w,x] = func_list[player](load)
            end
        end 
    elseif N == 20
        for i=1:dim[1], j=1:dim[2], k=1:dim[3], l=1:dim[4], m=1:dim[5], o=1:dim[6], p=1:dim[7], q=1:dim[8], r=1:dim[9], s=1:dim[10], t=1:dim[11], u=1:dim[12], v=1:dim[13], w=1:dim[14], x=1:dim[15], i16=1:dim[16], i17=1:dim[17], i18=1:dim[18], i19=1:dim[19], i20=1:dim[20]
            for n=1:N
                player = [i,j,k,l,m,o,p,q,r,s,t,u,v,w,x,i16,i17,i18,i19,i20][n]
                load = count(x->(x==player), [i,j,k,l,m,o,p,q,r,s,t,u,v,w,x,i16,i17,i18,i19,i20])
                cost_tensors[n][i,j,k,l,m,o,p,q,r,s,t,u,v,w,x,i16,i17,i18,i19,i20] = func_list[player](load)
            end
        end    

    end

    return cost_tensors
end

