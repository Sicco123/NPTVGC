function bounded_log(x)
    return log(max(x, 1e-10))
end

function lik_cv(obj, pv)
    γ,ϵ  = pv
    #weights_vec = [weights!((i, obj.ssize), γ, obj.weights, "CVsmo") for i in 1:obj.ssize]
    h_lik = total_likelihoods!(obj.x, obj.y, obj.ssize, obj.lags, obj.lags, ϵ, γ)

    L = sum(bounded_log.(h_lik[2:end]))

    neg_likelihood = -L / obj.ssize

    return neg_likelihood
end



function total_likelihoods!(x, y, N::Int, m::Int, mmax::Int, ϵ::Float64, γ)
    
    mu = (2.0 * ϵ)
    Cy = zeros(Float64, N)
    Cxy = zeros(Float64, N)
    Cyz = zeros(Float64, N)
    Cxyz = zeros(Float64, N)
    h = zeros(Float64, N)

    # vector N : 1 0 1 : N 
    part_2 = [abs(i - (N-2)) for i in 0:2*(N-2)]
    part_2 = γ.^part_2

    mid = N -1

    for i = 2:N
        Cy[i] = Cxy[i] = Cyz[i] = Cxyz[i] = 0.0
    
        t = i - 1
        part_1 = ((1-γ) / (2*γ-γ^(t)-γ^(N-(t)+1)))

        w = part_1 .* part_2[mid-t+1:end-t+1]
        w[t] = 0.0
        w_sum = 0.0
        
        @inbounds for j = 2:N
            w_j = w[j - 1]
           
            w_sum += w_j
            if j != i
                disx = disy = 0.0
                disx = abs(x[i-1] - x[j-1])
                disy = abs(y[i-1] - y[j-1])

                if disy <= ϵ 
                
                    Cy[i] += w_j
                    disx_ϵ = disx <= ϵ 
                    disx_ϵ && (Cxy[i] += w_j)
                    disz = max(abs(y[i] - y[j]), disy)
                    if disz <= ϵ
                        Cyz[i] += w_j
                        disx_ϵ && (Cxyz[i] += w_j)
                    end
                end
            end
        end
      
        h[i] += 2.0 / mu * (Cxyz[i] + Cy[i] + Cxy[i] + Cyz[i]) /w_sum
    
    end

    return h
end
