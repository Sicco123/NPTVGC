

# define maximum function
max(a, b) = a > b ? a : b

function get_h_vec!(x, y, N::Int, m::Int, mmax::Int, ϵ::Float64)
    
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
                    Cy[i] += 1
                    disx <= ϵ && (Cxy[i] += 1)
                    disz = max(abs(y[i] - y[j]), disy)
                    if disz <= ϵ
                        Cyz[i] += 1
                        disx <= ϵ && (Cxyz[i] += 1)
                    end
                end
            end
        end
        Cy[i] /= N - mmax
        Cxy[i] /= N - mmax
        Cyz[i] /= N - mmax
        Cxyz[i] /= N - mmax
        h[i] += 2.0 / mu * (Cxyz[i] * Cy[i] - Cxy[i] * Cyz[i]) / 6.0
    
    end

    for i = mmax+1:N
        for j = mmax+1:N
            if j != i
                IYij = IXYij = IYZij = IXYZij = 0
                disx = disy = 0.0
                for s = 1:m
                    disx = max(abs(x[i-s] - x[j-s]), disx)
                end
                for s = 1:mmax
                    disy = max(abs(y[i-s] - y[j-s]), disy)
                end
                if disy <= ϵ
                    IYij = 1
                    disx <= ϵ && (IXYij = 1)
                    disz = max(abs(y[i] - y[j]), disy)
                    if disz <= ϵ
                        IYZij = 1
                        disx <= ϵ && (IXYZij = 1)
                    end
                end
                h[i] += 2.0 / mu * (Cxyz[j] * IYij + IXYZij * Cy[j] - Cxy[j] * IYZij - IXYij * Cyz[j]) / (6 * (N - mmax))
            end
        end
    end
    return h
end

function get_h_vec_weighted!(x, y, N::Int, m::Int, mmax::Int, ϵ::Float64, w)
    
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
                    dc = w[i][j]*1 # weighted density contributons
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
        # Cy[i] /= N - mmax
        # Cxy[i] /= N - mmax
        # Cyz[i] /= N - mmax
        # Cxyz[i] /= N - mmax
        h[i] += 2.0 / mu * (Cxyz[i] * Cy[i] - Cxy[i] * Cyz[i]) / 6.0
    
    end

    for i = mmax+1:N
        for j = mmax+1:N
            if j != i
                IYij = IXYij = IYZij = IXYZij = 0
                disx = disy = 0.0
                for s = 1:m
                    disx = max(abs(x[i-s] - x[j-s]), disx)
                end
                for s = 1:mmax
                    disy = max(abs(y[i-s] - y[j-s]), disy)
                end
                if disy <= ϵ
                    dc = w[i][j]*1 # weighted density contributons
                    IYij = dc
                    disx <= ϵ && (IXYij = dc)
                    disz = max(abs(y[i] - y[j]), disy)
                    if disz <= ϵ
                        IYZij = dc
                        disx <= ϵ && (IXYZij = dc)
                    end
                end
                h[i] += 2.0 / mu * (Cxyz[j] * IYij + IXYZij * Cy[j] - Cxy[j] * IYZij - IXYij * Cyz[j]) / (6.0)
            end
        end
    end
    return h
end

function get_h_vec_weighted!(x, y, N::Int, m::Int, mmax::Int, ϵ1::Float64,ϵ2::Float64,ϵ3::Float64, w)
    
    mu1 = (2.0 * ϵ1)^( 2*mmax)
    mu2 = (2.0 * ϵ2)^(m)
    mu3 = (2.0 * ϵ3)^(1 )
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
                if disy <= ϵ1
                    dc = w[i][j]*1 # weighted density contributons
                    Cy[i] += dc
                    disx <= ϵ2 && (Cxy[i] += dc)
                    disz = max(abs(y[i] - y[j]), disy)
                    if disz <= 3
                        Cyz[i] += dc
                        disx <= ϵ2 && (Cxyz[i] += dc)
                    end
                end
            end
        end
        # Cy[i] /= N - mmax
        # Cxy[i] /= N - mmax
        # Cyz[i] /= N - mmax
        # Cxyz[i] /= N - mmax
        h[i] += 2.0/(mu1*mu2*mu3) * (Cxyz[i] * Cy[i] - Cxy[i] * Cyz[i]) / 6.0
    
    end

    for i = mmax+1:N
        for j = mmax+1:N
            if j != i
                IYij = IXYij = IYZij = IXYZij = 0
                disx = disy = 0.0
                for s = 1:m
                    disx = max(abs(x[i-s] - x[j-s]), disx)
                end
                for s = 1:mmax
                    disy = max(abs(y[i-s] - y[j-s]), disy)
                end
                if disy <= ϵ1
                    dc = w[i][j]*1 # weighted density contributons
                    IYij = dc
                    disx <= ϵ2 && (IXYij = dc)
                    disz = max(abs(y[i] - y[j]), disy)
                    if disz <= ϵ3
                        IYZij = dc
                        disx <= ϵ2 && (IXYZij = dc)
                    end
                end
                h[i] += 2.0 / (mu1*mu2*mu3) * (Cxyz[j] * IYij + IXYZij * Cy[j] - Cxy[j] * IYZij - IXYij * Cyz[j]) / (6.0)
            end
        end
    end
    return h
end


function get_h_vec_cuda!(x, y, N::Int, m::Int, mmax::Int, ϵ::Float64)
    
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
                    Cy[i] += 1
                    disx <= ϵ && (Cxy[i] += 1)
                    disz = max(abs(y[i] - y[j]), disy)
                    if disz <= ϵ
                        Cyz[i] += 1
                        disx <= ϵ && (Cxyz[i] += 1)
                    end
                end
            end
        end
        Cy[i] /= N - mmax
        Cxy[i] /= N - mmax
        Cyz[i] /= N - mmax
        Cxyz[i] /= N - mmax
        h[i] += 2.0 / mu * (Cxyz[i] * Cy[i] - Cxy[i] * Cyz[i]) / 6.0
    
    end

    for i = mmax+1:N
        for j = mmax+1:N
            if j != i
                IYij = IXYij = IYZij = IXYZij = 0
                disx = disy = 0.0
                for s = 1:m
                    disx = max(abs(x[i-s] - x[j-s]), disx)
                end
                for s = 1:mmax
                    disy = max(abs(y[i-s] - y[j-s]), disy)
                end
                if disy <= ϵ
                    IYij = 1
                    disx <= ϵ && (IXYij = 1)
                    disz = max(abs(y[i] - y[j]), disy)
                    if disz <= ϵ
                        IYZij = 1
                        disx <= ϵ && (IXYZij = 1)
                    end
                end
                h[i] += 2.0 / mu * (Cxyz[j] * IYij + IXYZij * Cy[j] - Cxy[j] * IYZij - IXYij * Cyz[j]) / (6 * (N - mmax))
            end
        end
    end
    return h
end



function HAC_variance(h, N, m)
    K = floor(Int, sqrt(sqrt(N)))

    ohm = [1.0; 2.0 * (1.0 .- (1:K-1) / K)]
    cov = zeros(Float64, K)

    # Determine autocovariance of h[i]
    for k = 0:K-1
        cov[k+1] = sum(h[m+k:N] .* h[m:N-k])/(N - m - k)
    end

    VT2 = 9.0 * dot(ohm, cov)
    return VT2
end

function HAC_variance(h, N, m, w)
    K = floor(Int, sqrt(sqrt(N)))

    ohm = [1.0; 2.0 * (1.0 .- (1:K-1) / K)]
    cov = zeros(Float64, K)

    # Determine autocovariance of h[i]
    for k = 0:K-1
        cov[k+1] = sum(h[m+k:N] .* h[m:N-k].* w[m+k:N])/sum(w[m+k:N])
    end

    VT2 = 9.0 * dot(ohm, cov)
    return VT2
end


# function estimate_tv_tstats(obj, s1, s2 = nothing)
#     # test for y does not cause x

#     if s2 === nothing
#         s2 = s1
#     end 

#     # Pre-compute weights and the h vector outside the loop
#     weights_vec = [weights!((i, obj.ssize), obj.γ, obj.weights, obj.filter) for i in s1:s2]
    
    
#     h_vec = get_h_vec!(obj.x, obj.y, obj.ssize, obj.lags, obj.lags, obj.ϵ)

#     # Initialize the numerators and vars arrays
#     numerators = similar(h_vec)  # Using similar to allocate space
#     vars = similar(h_vec)

#     # Compute numerators and vars in a single loop
#     for (idx, weights) in enumerate(weights_vec)
#         numerators[idx] = sum(h_vec[obj.lags+1:end] .* weights[obj.lags+1:end])
#         h_vec_adjusted = h_vec .- numerators[idx]  # Adjust h_vec inplace if possible
#         vars[idx] = HAC_variance(h_vec_adjusted, obj.ssize, obj.lags, weights)
#     end

#     # Calculate t-values and p-values outside the loop
#     T2_TVALS = numerators .* sqrt(obj.ssize - obj.lags) ./ sqrt.(vars)
#     p_T2s = 1 .- cdf.(Normal(0, 1), T2_TVALS)

#     # Update object properties
#     obj.Tstats = T2_TVALS
#     obj.pvals = p_T2s
#     return
# end

function estimate_tv_tstats(obj, s1)
    # test for y does not cause x
    s2 = obj.ssize

    # Pre-compute weights and the h vector outside the loop
    weights_vec = [weights!((i, obj.ssize), obj.γ, obj.weights, obj.filter) for i in s1:s2]

    
    h_vec = get_h_vec_weighted!(obj.x, obj.y, obj.ssize, obj.lags, obj.lags, obj.ϵ..., weights_vec)
    # Initialize the numerators and vars arrays
    numerators = similar(h_vec)  # Using similar to allocate space
    vars = similar(h_vec)

    # Compute numerators and vars in a single loop
    for (idx, weights) in enumerate(weights_vec)
        numerators[idx] = sum(h_vec[obj.lags+1:end].*weights[obj.lags+1:end])#(obj.ssize - obj.lags)
        h_vec_adjusted = h_vec .- numerators[idx]  # Adjust h_vec inplace if possible
        vars[idx] = HAC_variance(h_vec_adjusted, obj.ssize, obj.lags, weights)
    end

    # Calculate t-values and p-values outside the loop
    T2_TVALS =  numerators .* sqrt(obj.ssize - obj.lags) ./ sqrt.(vars) #numerators .* sqrt(obj.ssize - obj.lags) ./ sqrt.(vars)
    p_T2s = 0.5 .- 0.5.*cdf.(Normal(0, 1), T2_TVALS./sqrt(2)) # 1 .- cdf.(Normal(0, 1), T2_TVALS) #

    # Update object properties
    obj.Tstats = T2_TVALS
    obj.pvals = p_T2s
    return
end

function estimate_constant_tstat(obj, s1)
    # test for y does not cause x
    s2 = obj.ssize

    # Pre-compute weights and the h vector outside the loop
    weights_vec = [weights!((i, obj.ssize), obj.γ, obj.weights, obj.filter) for i in s1:s2]

    
    h_vec = get_h_vec_weighted!(obj.x, obj.y, obj.ssize, obj.lags, obj.lags, obj.ϵ..., weights_vec)
    # Initialize the numerators and vars arrays
    numerators = similar(h_vec)  # Using similar to allocate space
    vars = similar(h_vec)

    # Compute numerators and vars in a single loop
    numerator = sum(h_vec[obj.lags+1:end])/(obj.ssize - obj.lags)
    h_vec_adjusted = h_vec .- numerators  # Adjust h_vec inplace if possible
    var = HAC_variance(h_vec_adjusted, obj.ssize, obj.lags)
    

    # Calculate t-values and p-values outside the loop
    T2_TVAL =  numerator .* sqrt(obj.ssize - obj.lags) ./ sqrt(var) #numerators .* sqrt(obj.ssize - obj.lags) ./ sqrt.(vars)
    p_T2 = 0.5 - 0.5*cdf(Normal(0, 1), T2_TVAL/sqrt(2)) # 1 .- cdf.(Normal(0, 1), T2_TVALS) #

    # Update object properties
    obj.Tstats = [T2_TVAL]
    obj.pvals = [p_T2]
    return
end