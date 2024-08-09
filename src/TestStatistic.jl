

function get_h_matrix_weighted!(x, y,  N::Int, γ, ϵ::Float64  )

    Ay, Axy, Ayz, Axyz = get_indicator_matrices!(x, y, N, ϵ)
    Cy, Cxy, Cyz, Cxyz = sum(Ay, dims = 2), sum(Axy, dims = 2), sum(Ayz, dims = 2), sum(Axyz, dims = 2)

    Ay, Axy, Ayz, Axyz = @view(Ay[2:end,2:end]), @view(Axy[2:end,2:end]), @view(Ayz[2:end,2:end]), @view(Axyz[2:end,2:end])
    Cy, Cxy, Cyz, Cxyz = @view(Cy[2:end]), @view(Cxy[2:end]), @view(Cyz[2:end]), @view(Cxyz[2:end])
    
    interactions = Matrix{Float64}(undef, size(Ay, 1), N-1)  # Preallocate matrix

    for jdx in 1:N-1
        interactions[:, jdx] = (Cxyz[jdx] .* Ay[:,jdx]  
        + Axyz[:,jdx] .* Cy[jdx] 
            - Cxy[jdx] .* Ayz[:,jdx] 
            - Axy[:,jdx] .* Cyz[jdx])
    end

    μ = (2.0 * ϵ)^(4)

    h = zeros(Float64, N, N)

    # Weight calcualtion 
    # vector N : 1 0 1 : N 
    weight_reduction = [abs(i - (N-2)) for i in 0:2*(N-2)]
    weight_reduction = γ.^weight_reduction

    mid = N -1

    h_col_vec = zeros(Float64, N-1)
    factor = zeros(Float64, N-1)
    w = zeros(Float64, N-1) 

    for t in 1:N-1
        part_1 = ((1-γ) / (1+γ-γ^(t)-γ^(N-t)))
        w = part_1 .* weight_reduction[mid-t+1:end-t+1]
        
        h[2:end, t+1] +=  2.0/μ .* w.^2 .* (Cxyz .* Cy .- Cxy .* Cyz)./6.0
        
        factor = 2.0 / μ  .* w /6.0
        h_col_vec = sum(  interactions .* w', dims=2)  # Vectorized sum of interactions
        h[2:end, t+1] += factor .* h_col_vec
        
    end
    return h
end 

function get_indicator_matrices!(x, y, N::Int,   ϵ::Float64 )

    Cy = zeros(Float64, N, N)
    Cxy = zeros(Float64, N, N)
    Cyz = zeros(Float64, N, N)
    Cxyz = zeros(Float64, N, N)

    for i = 2:N
        for j = 2:N      
            if j != i
                disx = disy = 0.0
                disx = abs(x[i-1] - x[j-1])
                disy = abs(y[i-1] - y[j-1])
            
                if disy <= ϵ      
                    Cy[i,j] += 1
                    disx_ϵ = disx <= ϵ 
                    disx_ϵ && (Cxy[i,j] += 1)
                    disz = max(abs(y[i] - y[j]), disy)
                    if disz <= ϵ
                        Cyz[i,j] += 1
                        disx_ϵ && (Cxyz[i,j] += 1)
                    end
                end
            end
        end
    end
    return Cy, Cxy, Cyz, Cxyz
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

function estimate_tv_tstats(obj, s1)
    # test for y does not cause x
    s2 = obj.ssize

    h_matrix = get_h_matrix_weighted!(obj.x, obj.y, obj.ssize, obj.γ, obj.ϵ...)
    
    h_column_means = mean(h_matrix, dims = 1)
   
    demeaned_h_matrix = h_matrix .- h_column_means
    

    vars = similar(h_column_means)
    vars = mapslices(x -> HAC_variance(x, obj.ssize, obj.lags), demeaned_h_matrix; dims = 1)

    # Calculate t-values and p-values outside the loop
    T2_TVALS =  h_column_means .* sqrt(obj.ssize - obj.lags) ./ sqrt.(vars) #numerators .* sqrt(obj.ssize - obj.lags) ./ sqrt.(vars)
    p_T2s = 0.5 .- 0.5.*cdf.(Normal(0, 1), T2_TVALS./sqrt(2)) # 1 .- cdf.(Normal(0, 1), T2_TVALS) #


    # Update object properties
    obj.Tstats = T2_TVALS[1,:]
    obj.pvals = p_T2s[1,:]
    return
end
