function bounded_log(x)
    return log(max(x, 1e-10))
end

function lik_cv(obj, pv)
    γ,ϵ  = pv

    weights_vec = [weights!((i, obj.ssize), γ, obj.weights, "CVsmo") for i in 1:obj.ssize]
    h_lik = total_likelihoods!(obj.x, obj.y, obj.ssize, obj.lags, obj.lags, ϵ, weights_vec)
   
    L = sum(bounded_log.(h_lik[2:end]))

    neg_likelihood = -L / obj.ssize
    return neg_likelihood
end


function total_likelihoods!(x, y, N::Int, m::Int, mmax::Int, ϵ::Float64, w)
    
    mu = (2.0 * ϵ)^(m + 2 * mmax + 1)
    Cy = zeros(Float64, N)
    Cxy = zeros(Float64, N)
    Cyz = zeros(Float64, N)
    Cxyz = zeros(Float64, N)
    h = zeros(Float64, N)

    for i = mmax+1:N
        Cy[i] = Cxy[i] = Cyz[i] = Cxyz[i] = 0.0
        for j = mmax+1:N
            if j != i
                disx = disy = 0.0
                for s = 1:m
                    disx = max(abs(x[i-s] - x[j-s]), disx)
                end
                for s = 1:mmax
                    disy = max(abs(y[i-s] - y[j-s]), disy)
                end
                if disy <= ϵ
                    dc = w[i][j]*1
                    Cy[i] += dc
                    disx <= ϵ && (Cxy[i] += dc)
                    disz = max(abs(y[i] - y[j]), disy)
                    if disz <= ϵ
                        Cyz[i] += dc
                        disx <= ϵ && (Cxyz[i] += dc)
                    end
                end
            end
        end
      
        h[i] += 2.0 / mu * (Cxyz[i] + Cy[i] + Cxy[i] + Cyz[i]) 
    
    end

    return h
end
