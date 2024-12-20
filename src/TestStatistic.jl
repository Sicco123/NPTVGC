function get_h_matrix!(x, y,  N::Int, γ, ϵ::Float64  )

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

    h[2:end, 2] +=  2.0/μ .* (1/(N-1)).^2 .* (Cxyz .* Cy .- Cxy .* Cyz)./6.0
    
    factor = 2.0 / μ  .* (1/(N-1)) /6.0
    h_col_vec = sum(interactions, dims=2)./(N-1)  # Vectorized sum of interactions
    h[2:end, 2] += factor .* h_col_vec
    
    # copy the column to the rest of the columns
    for t in 3:N
        h[2:end, t] = h[2:end, 2]
    end
    
    return h
end 

# function get_h_matrix_weighted!(x, y,  N::Int, γ, ϵ::Float64  )

#     Ay, Axy, Ayz, Axyz = get_indicator_matrices!(x, y, N, ϵ)
#     Cy, Cxy, Cyz, Cxyz = sum(Ay, dims = 2), sum(Axy, dims = 2), sum(Ayz, dims = 2), sum(Axyz, dims = 2)

#     Ay, Axy, Ayz, Axyz = @view(Ay[2:end,2:end]), @view(Axy[2:end,2:end]), @view(Ayz[2:end,2:end]), @view(Axyz[2:end,2:end])
#     Cy, Cxy, Cyz, Cxyz = @view(Cy[2:end]), @view(Cxy[2:end]), @view(Cyz[2:end]), @view(Cxyz[2:end])
    
#     interactions = Matrix{Float64}(undef, size(Ay, 1), N-1)  # Preallocate matrix

#     for jdx in 1:N-1
#         interactions[:, jdx] = (Cxyz[jdx] .* Ay[:,jdx]  
#         + Axyz[:,jdx] .* Cy[jdx] 
#             - Cxy[jdx] .* Ayz[:,jdx] 
#             - Axy[:,jdx] .* Cyz[jdx])
#     end

#     μ = (2.0 * ϵ)^(4)

#     h = zeros(Float64, N, N)

#     # Weight calcualtion 
#     # vector N : 1 0 1 : N 
#     weight_reduction = [abs(i - (N-2)) for i in 0:2*(N-2)]
#     weight_reduction = γ.^weight_reduction

#     mid = N -1

#     h_col_vec = zeros(Float64, N-1)
#     factor = zeros(Float64, N-1)
#     w = zeros(Float64, N-1) 

#     for t in 1:N-1
#         part_1 = ((1-γ) / (1+γ-γ^(t)-γ^(N-t)))
#         w = part_1 .* weight_reduction[mid-t+1:end-t+1]
        
#         h[2:end, t+1] +=  2.0/μ .* w.^2 .* (Cxyz .* Cy .- Cxy .* Cyz)./6.0
        
#         factor = 2.0 / μ  .* w /6.0
#         h_col_vec = sum(  interactions .* w', dims=2)  # Vectorized sum of interactions
#         h[2:end, t+1] += factor .* h_col_vec
        
#     end
#     return h
# end 

# function get_h_matrix_weighted!(x, y,  N::Int, γ, ϵ::Float64  )

#     Ay, Axy, Ayz, Axyz = get_indicator_matrices!(x, y, N, ϵ)
#     Cy, Cxy, Cyz, Cxyz = sum(Ay, dims = 2), sum(Axy, dims = 2), sum(Ayz, dims = 2), sum(Axyz, dims = 2)

#     Ay, Axy, Ayz, Axyz = @view(Ay[2:end,2:end]), @view(Axy[2:end,2:end]), @view(Ayz[2:end,2:end]), @view(Axyz[2:end,2:end])
#     Cy, Cxy, Cyz, Cxyz = @view(Cy[2:end]), @view(Cxy[2:end]), @view(Cyz[2:end]), @view(Cxyz[2:end])
    
#     interactions = Matrix{Float64}(undef, size(Ay, 1), N-1)  # Preallocate matrix

#     for jdx in 1:N-1
#         interactions[:, jdx] = (Cxyz[jdx] .* Ay[:,jdx]  
#         + Axyz[:,jdx] .* Cy[jdx] 
#             - Cxy[jdx] .* Ayz[:,jdx] 
#             - Axy[:,jdx] .* Cyz[jdx])
#     end

#     μ = (2.0 * ϵ)^(4)

#     h = zeros(Float64, N, N)

#     # Weight calcualtion 
#     # vector N : 1 0 1 : N 
#     weight_reduction = [abs(i - (N-2)) for i in 0:2*(N-2)]
#     weight_reduction = γ.^weight_reduction

#     mid = N -1

#     h_col_vec = zeros(Float64, N-1)
#     factor = zeros(Float64, N-1)
#     w = zeros(Float64, N-1) 

#     for t in 1:N-1
#         part_1 = ((1-γ) / (1+γ-γ^(t)-γ^(N-t)))
#         w = part_1 .* weight_reduction[mid-t+1:end-t+1]
        
#         h[2:end, t+1] +=  2.0/μ .* w.^2 .* (Cxyz .* Cy .- Cxy .* Cyz)./6.0
        
#         factor = 2.0 / μ  .* w /6.0
#         h_col_vec = sum(  interactions .* w', dims=2)  # Vectorized sum of interactions
#         h[2:end, t+1] += factor .* h_col_vec
        
#     end
#     return h
# end 




# # Define kernel function
# function h_kernel_func!(h, interactions,  weight_reduction, Cxyz, Cy, Cxy, Cyz, γ, μ, N, mid, threshold) #interactions,
#     idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x - 1

#     if idx < (N - 1)^2  # Ensure we are within bounds
#         i = div(idx, N - 1) + 1  # Compute the row index
#         t = mod(idx, N - 1) + 1  # Compute the column index

#         index = mid - t + i
#         if weight_reduction[index] > threshold
#             # Precompute part_1 to avoid redundant calculations
#             γ_t = γ^t
#             γ_n_t = γ^(N - t)
#             part_1 = (1 - γ) / (1 + γ - γ_t - γ_n_t)

#             w_i = part_1 * weight_reduction[index]
            
#             # Accumulate h[i+1, t+1] updates in a local variable
#             @views local_h_update = (2.0 / μ) * w_i^2 * (Cxyz[i] * Cy[i] - Cxy[i] * Cyz[i]) / 6.0
            
#             # Precompute factor_i
#             factor_i = (2.0 / μ) * w_i / 6.0

#             # Accumulate h_col_vec_i in local variable to minimize memory access
#             h_col_vec_i = 0.0
#             jdx_0 = mid - t 
#             for j in 1:N - 1
#                 @inbounds h_col_vec_i += weight_reduction[jdx_0+j]*interactions[i, j]  * part_1 
#             end
            
#             # Update h with accumulated values
#             h[i + 1, t + 1] += (local_h_update + factor_i * h_col_vec_i) 
            
#         end
#     end
#     return
# end

# function get_h_matrix_weighted_cuda!(x, y,  N::Int, γ, ϵ::Float64  )

#     Ay, Axy, Ayz, Axyz = get_indicator_matrices!(x, y, N, ϵ)
#     Cy, Cxy, Cyz, Cxyz = sum(Ay, dims = 2), sum(Axy, dims = 2), sum(Ayz, dims = 2), sum(Axyz, dims = 2)

#     Ay, Axy, Ayz, Axyz = @view(Ay[2:end,2:end]), @view(Axy[2:end,2:end]), @view(Ayz[2:end,2:end]), @view(Axyz[2:end,2:end])
#     Cy, Cxy, Cyz, Cxyz = @view(Cy[2:end]), @view(Cxy[2:end]), @view(Cyz[2:end]), @view(Cxyz[2:end])
    
#     interactions = Matrix{Float64}(undef, size(Ay, 1), N-1)  # Preallocate matrix

#     for jdx in 1:N-1
#         interactions[:, jdx] = (Cxyz[jdx] .* Ay[:,jdx]  
#         + Axyz[:,jdx] .* Cy[jdx] 
#             - Cxy[jdx] .* Ayz[:,jdx] 
#             - Axy[:,jdx] .* Cyz[jdx])
#     end

#     μ = (2.0 * ϵ)^(4)

#     h = zeros(Float64, N, N)

#     # Weight calcualtion 
#     # vector N : 1 0 1 : N 
#     weight_reduction = [abs(i - (N-2)) for i in 0:2*(N-2)]
#     weight_reduction = γ.^weight_reduction

#     mid = N -1

#     h_col_vec = zeros(Float64, N-1)
#     factor = zeros(Float64, N-1)
    
#     # Device arrays for GPU computation
#     d_h = CuArray{Float32}(h)
#     d_interactions = CuArray{Float32}(interactions)
#     d_weight_reduction = CuArray{Float32}(weight_reduction)
#     d_Cxyz = CuArray{Float32}(Cxyz)
#     d_Cy = CuArray{Float32}(Cy)
#     d_Cxy = CuArray{Float32}(Cxy)
#     d_Cyz = CuArray{Float32}(Cyz)

#     d_h_col_vec = CUDA.zeros(Float32, N-1)
#     d_factor = CUDA.zeros(Float32, N-1)
#     d_w = CUDA.zeros(Float32, N-1)

#     threads = 64 #128 #576#threads
#     blocks = ceil(Int64, (N-1)^2/threads) 
#     threshold = 10^-8
#     @cuda threads=threads blocks=blocks h_kernel_func!(d_h,d_interactions, d_weight_reduction, d_Cxyz, d_Cy, d_Cxy, d_Cyz, γ, μ, N, mid, threshold) 

#     CUDA.synchronize()

#     # Copy result back to host
#     h .= Array(d_h)


# end 

# function get_indicator_matrices!(x, y, N::Int,   ϵ::Float64 )

#     Cy = zeros(Float64, N, N)
#     Cxy = zeros(Float64, N, N)
#     Cyz = zeros(Float64, N, N)
#     Cxyz = zeros(Float64, N, N)

#     for i = 2:N
#         for j = 2:N      
#             if j != i
#                 disx = disy = 0.0
#                 disx = abs(x[i-1] - x[j-1])
#                 disy = abs(y[i-1] - y[j-1])
            
#                 if disy <= ϵ      
#                     Cy[i,j] += 1
#                     disx_ϵ = disx <= ϵ 
#                     disx_ϵ && (Cxy[i,j] += 1)
#                     disz = max(abs(y[i] - y[j]), disy)
#                     if disz <= ϵ
#                         Cyz[i,j] += 1
#                         disx_ϵ && (Cxyz[i,j] += 1)
#                     end
#                 end
#             end
#         end
#     end
#     return Cy, Cxy, Cyz, Cxyz
# end


# function HAC_variance(h, N, m)
#     K = floor(Int, sqrt(sqrt(N)))

#     ohm = [1.0; 2.0 * (1.0 .- (1:K-1) / K)]
#     cov = zeros(Float64, K)


#     # Determine autocovariance of h[i]
#     for k in 1:K
#         cov[k] = 0.0
#         for i in (m+k):N
#             cov[k] += h[i] * h[i-k+1]  # +1 for Julia's 1-based indexing
#         end
#         cov[k] /= (N - m - k +1)  # +1 for correct denominator in 1-based indexing
#     end

#     VT2 = 9.0 * dot(ohm, cov)
#     return VT2
# end


# function estimate_tv_tstats(obj; device = "cpu")
#     # test for y does not cause x
#     s2 = obj.ssize

#     if obj.γ == 1.0
#         estimate_constant_tstats(obj)
#         return
#     end

#     if device == "cuda"
#         h_matrix = get_h_matrix_weighted_cuda!(obj.x, obj.y, obj.ssize, obj.γ, obj.ϵ...)
#     else 
#         h_matrix = get_h_matrix_weighted!(obj.x, obj.y, obj.ssize, obj.γ, obj.ϵ...)
#     end
#     h_column_means = mean(h_matrix, dims = 1) * obj.ssize/(obj.ssize - obj.lags)
   
#     demeaned_h_matrix = h_matrix .- h_column_means
    

#     vars = similar(h_column_means)
#     vars = mapslices(x -> HAC_variance(x, obj.ssize, obj.lags), demeaned_h_matrix; dims = 1)

#     # Calculate t-values and p-values outside the loop
#     T2_TVALS =  h_column_means .* sqrt(obj.ssize - obj.lags) ./ sqrt.(vars) #numerators .* sqrt(obj.ssize - obj.lags) ./ sqrt.(vars)
#     p_T2s = 1 .- cdf.(Normal(0, 1), T2_TVALS) #0.5 .- 0.5.*cdf.(Normal(0, 1), T2_TVALS./sqrt(2)) # 


#     # Update object properties
#     obj.Tstats = T2_TVALS[1,:]
#     obj.pvals = p_T2s[1,:]
#     return
# end


function estimate_constant_tstats(obj)
    h_matrix = get_h_matrix!(obj.x, obj.y, obj.ssize, obj.γ, obj.ϵ...)
    
    h_column_means = mean(h_matrix, dims = 1) * obj.ssize/(obj.ssize - obj.lags)
 
    demeaned_h_matrix = h_matrix .- h_column_means
    

    vars = similar(h_column_means)
    vars = mapslices(x -> HAC_variance_orig(x, obj.ssize, obj.lags), demeaned_h_matrix; dims = 1)


    # Calculate t-values and p-values outside the loop
    T2_TVALS =  h_column_means .* sqrt(obj.ssize - obj.lags) ./ sqrt.(vars) #numerators .* sqrt(obj.ssize - obj.lags) ./ sqrt.(vars)
    p_T2s = 1 .- cdf.(Normal(0, 1), T2_TVALS) #0.5 .- 0.5.*cdf.(Normal(0, 1), T2_TVALS./sqrt(2)) # 


    # Update object properties
    obj.Tstats = T2_TVALS[1,:]
    obj.pvals = p_T2s[1,:]
    return
end 


function estimate_tv_tstats(obj; device = "CPU")
    if obj.γ == 1.0
        estimate_constant_tstats(obj)
        return
    end

    if device == "CUDA"
        estimate_tv_tstats_cuda(obj)
    else
        estimate_tv_tstats_cpu(obj)
    end
end

function estimate_tv_tstats_cpu(obj)

    x = obj.x
    y = obj.y
    N = obj.ssize
    γ = obj.γ
    ϵ = obj.ϵ[1]
    lags = obj.lags

    # Pre-allocate output arrays
    n_tests = N-1
    T2_TVALS = Vector{Float32}(undef, n_tests)
    p_T2s = Vector{Float32}(undef, n_tests)
    
    # Pre-calculate weight reduction array once
    mid = N-1
    weight_reduction = Vector{Float32}(undef, 2*(N-2)+1)
    @inbounds for i in 0:2*(N-2)
        weight_reduction[i+1] = Float32(γ^abs(i - (N-2)))
    end
    
    # Pre-allocate reusable arrays
    Cy = zeros(Float32, N)
    Cxy = zeros(Float32, N)
    Cyz = zeros(Float32, N)
    Cxyz = zeros(Float32, N)
    h = zeros(Float32, N)
    
    # Pre-allocate arrays for intermediate calculations
    h_weighted = Vector{Float32}(undef, N-1)
    h_vec_adjusted = Vector{Float32}(undef, N-1)
    
    # Define threshold for "effectively zero"
    weight_threshold = Float32(1e-9)  # Adjusted for Float32 precision
    
    @inbounds for t in 1:n_tests
        # Calculate w_sum more efficiently
        w_sum = Float32((1 + γ - γ^t - γ^(N-t)) / (1-γ))
        
        # Get window of weights without creating new array
        w_start = mid-t+1
        w_end = length(weight_reduction)-t+1
        
        # Reset arrays
        fill!(Cy, 0.0f0)
        fill!(Cxy, 0.0f0)
        fill!(Cyz, 0.0f0)
        fill!(Cxyz, 0.0f0)
        fill!(h, 0.0f0)
        
        # Calculate h_weighted using weight_reduction window
        get_h_vec_weighted_efficient!(x, y, N, Float32(ϵ), view(weight_reduction, w_start:w_end), 
                                    w_sum, Cy, Cxy, Cyz, Cxyz, h, weight_threshold)
      
        # Calculate weighted h values
        @inbounds for i in 1:N-1
            h_weighted[i] = h[i+1] * weight_reduction[w_start+i-1]
        end
       
        
        # Calculate numera more efficiently
        numera = sum(h_weighted)/w_sum

        # Adjust h_vec in place
        @inbounds for i in 1:N-1
            h_vec_adjusted[i] = h_weighted[i] - numera
        end

        # Calculate variance and test statistics
        var = HAC_variance_orig(h_vec_adjusted, N-1, 1)
   
        T2_TVALS[t] = numera * sqrt(Float32(N - 1)) / sqrt(var)
        p_T2s[t] = 1 - cdf(Normal(0, 1), T2_TVALS[t])
    end
    
    obj.Tstats = T2_TVALS
    obj.pvals = p_T2s
    return 
end

function get_h_vec_weighted_efficient!(x, y, N::Int, ϵ::Float32, w::SubArray{Float32}, 
                                     w_sum::Float32, Cy::Vector{Float32}, Cxy::Vector{Float32}, 
                                     Cyz::Vector{Float32}, Cxyz::Vector{Float32}, h::Vector{Float32},
                                     weight_threshold::Float32)
    mu = Float32((2.0 * ϵ)^4)
    
    # First pass - calculate C values
    @inbounds for i = 2:N
        for j = 2:N
            if j != i
                # Skip iteration if weight is effectively zero
                if w[j-1] <= weight_threshold
                    continue
                end
                
                disx = abs(x[i-1] - x[j-1])
                disy = abs(y[i-1] - y[j-1])
                
                if disy <= ϵ
                    Cy[i] += w[j-1]
                    disx <= ϵ && (Cxy[i] += w[j-1])
                    disz = abs(y[i] - y[j])
                    if disz <= ϵ
                        Cyz[i] += w[j-1]
                        disx <= ϵ && (Cxyz[i] += w[j-1])
                    end
                end
            end
        end
        
        # Normalize C values
        inv_w_sum = 1.0f0 / w_sum
        Cy[i] *= inv_w_sum
        Cxy[i] *= inv_w_sum
        Cyz[i] *= inv_w_sum
        Cxyz[i] *= inv_w_sum
    end
    
    # Second pass - calculate h values
    inv_mu = 2.0f0 / mu
    IYij = IXYij = IYZij = IXYZij = 0.0f0
   
    @inbounds for i = 2:N
        h_base = (Cxyz[i] * Cy[i] - Cxy[i] * Cyz[i]) / 6.0f0
        h[i] += inv_mu * h_base
        
        for j = 2:N
            if j != i
                # Skip iteration if weight is effectively zero
                if w[j-1] <= weight_threshold
                    continue
                end
                
                IYij = IXYij = IYZij = IXYZij = 0.0f0
                disx = abs(x[i-1] - x[j-1])
                disy = abs(y[i-1] - y[j-1])
                
                if disy <= ϵ
                    IYij = w[j-1]
                    IXYij = disx <= ϵ ? w[j-1] : 0.0f0
                    disz = abs(y[i] - y[j])
                    
                    if disz <= ϵ
                        IYZij = w[j-1]
                        IXYZij = disx <= ϵ ? w[j-1] : 0.0f0
                    end
                end
                h[i] += inv_mu * (Cxyz[j] * IYij + IXYZij * Cy[j] - 
                Cxy[j] * IYZij - IXYij * Cyz[j]) / (6.0f0 * w_sum)
            end
        end
    end
    
    return h
end

function HAC_variance_orig(h, N, m)
    K = floor(Int, sqrt(sqrt(N)))

    ohm = Float32[1.0f0; 2.0f0 * (1.0f0 .- (1:K-1) / K)]
    cov = zeros(Float32, K)

    # Determine autocovariance of h[i]
    @inbounds for k in 1:K
        cov[k] = 0.0f0
        for i in (m+k):N
            cov[k] += h[i] * h[i-k+1]  # +1 for Julia's 1-based indexing
        end
        cov[k] /= (N - m - k +1)  # +1 for correct denominator in 1-based indexing
    end

    VT2 = 9.0f0 * dot(ohm, cov)

    return VT2
end
