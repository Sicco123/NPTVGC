

function get_h_stats_incremental!(x, y, N::Int, γ, ϵ::Float64)
    # Pre-allocate indicator matrices and reuse them
    Ay = similar(x, N, N)
    Axy = similar(x, N, N)
    Ayz = similar(x, N, N)
    Axyz = similar(x, N, N)
    
    # Get indicator matrices
    Ay, Axy, Ayz, Axyz = get_indicator_matrices!(x, y, N, ϵ)

    # Compute sums using views
    Cy = vec(sum(view(Ay, :, :), dims=2))
    Cxy = vec(sum(view(Axy, :, :), dims=2))
    Cyz = vec(sum(view(Ayz, :, :), dims=2))
    Cxyz = vec(sum(view(Axyz, :, :), dims=2))

    # Create views
    Ay_view = view(Ay, 2:N, 2:N)
    Axy_view = view(Axy, 2:N, 2:N)
    Ayz_view = view(Ayz, 2:N, 2:N)
    Axyz_view = view(Axyz, 2:N, 2:N)
    
    Cy_view = view(Cy, 2:N)
    Cxy_view = view(Cxy, 2:N)
    Cyz_view = view(Cyz, 2:N)
    Cxyz_view = view(Cxyz, 2:N)

    # Precompute constants
    μ = (2.0 * ϵ)^4
    μ_factor = 2.0 / (μ * 6.0)
    mid = N - 1
    
    # Initialize output statistics vectors
    h_means = zeros(Float64, N)
    vars = zeros(Float64, N)
    
    # Pre-allocate arrays for intermediate calculations
    interactions = zeros(Float64, N-1, N-1)
    w = Vector{Float64}(undef, N-1)
    h_column = Vector{Float64}(undef, N)  # Store one column at a time, full size for HAC
    
    # Pre-allocate weight reduction vector
    weight_reduction = [γ^abs(i - (N-2)) for i in 0:2*(N-2)]
    
    # Compute interactions matrix
    @inbounds for jdx in 1:N-1
        @simd for idx in 1:N-1
            interactions[idx, jdx] = (Cxyz_view[jdx] * Ay_view[idx,jdx] + 
                                    Axyz_view[idx,jdx] * Cy_view[jdx] - 
                                    Cxy_view[jdx] * Ayz_view[idx,jdx] - 
                                    Axy_view[idx,jdx] * Cyz_view[jdx])
        end
    end
    
    # Process one column at a time
    @inbounds for t in 1:N-1
        # Reset h_column for this iteration
        fill!(h_column, 0.0)
        
        # Compute weights
        part_1 = (1-γ) / (1+γ-γ^t-γ^(N-t))
        @simd for i in 1:N-1
            w[i] = part_1 * weight_reduction[mid-t+i]
        end
        
        # Compute h values for current column
        @simd for i in 1:N-1
            # Base terms
            base_term = μ_factor * w[i]^2 * (Cxyz_view[i] * Cy_view[i] - Cxy_view[i] * Cyz_view[i])
            
            # Interaction terms
            temp_sum = 0.0
            for j in 1:N-1
                temp_sum += interactions[i,j] * w[j]
            end
            interaction_term = μ_factor * w[i] * temp_sum
            
            # Store current h value
            h_column[i+1] = (base_term + interaction_term) * w[i]
        end
        
        # Calculate mean for this column
        h_means[t+1] = sum(h_column)
        
        # Calculate HAC variance for this column
        h_column .*= N  # Scale by N as in original code
        h_column .-= h_means[t+1]  # Demean
        vars[t+1] = HAC_variance(h_column, N, 1)
    end
    
    return h_means, vars
end

function estimate_tv_tstats_boosted_memory(obj; device = "cpu")
    s2 = obj.ssize

    if obj.γ == 1.0
        estimate_constant_tstats(obj)
        return
    end

    # Get statistics without storing full matrix
    if device == "cuda"
        h_means, vars = get_h_stats_incremental_cuda!(obj.x, obj.y, obj.ssize, obj.γ, obj.ϵ...)
    else 
        h_means, vars = get_h_stats_incremental!(obj.x, obj.y, obj.ssize, obj.γ, obj.ϵ...)
    end
    
    # Adjust means
    factor = obj.ssize/(obj.ssize - obj.lags)
    h_means .*= factor

    # Calculate t-values and p-values
    T2_TVALS = h_means .* sqrt(obj.ssize - obj.lags) ./ sqrt.(vars)
    p_T2s = 1.0 .- cdf.(Normal(0, 1), T2_TVALS)

    # Update object properties
    obj.Tstats = T2_TVALS
    obj.pvals = p_T2s
    return
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

