function estimate_tv_tstats_cuda(obj)
    x = obj.x
    y = obj.y
    N = length(x)
    ϵ = obj.ϵ
    γ = obj.γ


    # convert to Float32
    x_d = CuArray{Float32}(x)
    y_d = CuArray{Float32}(y)
    ϵ = Float32(ϵ)
    γ = Float32(γ)

    
    # Pre-allocate output arrays on CPU (will transfer results back)
    n_tests = N-1
    T2_TVALS = Vector{Float32}(undef, n_tests)
    p_T2s = Vector{Float32}(undef, n_tests)
    
    # Pre-calculate weight reduction array once
    mid = N-1
    weight_reduction = Vector{Float32}(undef, 2*(N-2)+1)
    @inbounds for i in 0:2*(N-2)
        weight_reduction[i+1] = Float32(γ^abs(i - (N-2)))
    end
    weight_reduction = CuArray{Float32}(weight_reduction)

    # Pre-allocate reusable arrays on GPU
    Cy = CUDA.zeros(Float32, N)
    Cxy = CUDA.zeros(Float32, N)
    Cyz = CUDA.zeros(Float32, N)
    Cxyz = CUDA.zeros(Float32, N)
    h = CUDA.zeros(Float32, N)
    
    # Pre-allocate intermediate arrays on GPU
    h_weighted = CUDA.zeros(Float32, N-1)
    h_vec_adjusted = CUDA.zeros(Float32, N-1)
    
    # Define threshold
    weight_threshold = Float32(1e-6)
 
    
    for t in 1:n_tests
        # Calculate w_sum
        w_sum = Float32((1 + γ - γ^t - γ^(N-t)) / (1-γ))
        
        # Get window indices
        w_start = mid-t+1
        w_end = length(weight_reduction)-t+1
        
        # Reset arrays
        CUDA.fill!(Cy, 0.0f0)
        CUDA.fill!(Cxy, 0.0f0)
        CUDA.fill!(Cyz, 0.0f0)
        CUDA.fill!(Cxyz, 0.0f0)
        CUDA.fill!(h, 0.0f0)
        
  
        # Calculate optimal thread and block counts
        threads_per_block = 128  # Should be power of 2 for reduction
        blocks =  min(N - 1, 65535)  # One block per i value (excluding i=1)
        shared_mem_size = 4 * threads_per_block * sizeof(Float32)  # Five arrays for reductions
        
        # Launch kernel with adjusted parameters
        @cuda blocks=blocks threads=threads_per_block shmem=shared_mem_size get_h_vec_kernel!(
            x_d, y_d, N, ϵ, view(weight_reduction, w_start:w_end),
            w_sum, Cy, Cxy, Cyz, Cxyz, h, weight_threshold
        )

        CUDA.synchronize()

        shared_mem_size = threads_per_block * sizeof(Float32)  # Five arrays for reductions
        
        # Add cross terms 
        @cuda blocks=blocks threads=threads_per_block shmem=shared_mem_size get_h_vec_cross_terms_kernel!(
        x_d, y_d, N, ϵ, view(weight_reduction, w_start:w_end),
        w_sum, Cy, Cxy, Cyz, Cxyz, h, weight_threshold
        )

        CUDA.synchronize()

        # Calculate weighted h values
        @cuda threads=128 blocks=ceil(Int, (N-1)/128) compute_h_weighted_kernel!(
            h_weighted, h, view(weight_reduction, w_start:w_end), N
        )

        CUDA.synchronize()
        # Calculate numera using reduction
        numera = CUDA.reduce(+,h_weighted)/w_sum

        # Adjust h_vec in parallel
        @cuda threads=128 blocks=ceil(Int, (N-1)/128) adjust_h_vec_kernel!(
            h_vec_adjusted, h_weighted, numera, N-1
        )

        CUDA.synchronize()

        # move h_vec_adjusted to CPU
        h_vec_adjusted_cpu = Array(h_vec_adjusted)
        
        # Calculate variance using HAC
        var = HAC_variance_orig(h_vec_adjusted_cpu, N-1, 1) #HAC_variance_cuda(h_vec_adjusted, N-1, 1)


        # Calculate final statistics on CPU
        T2_TVALS[t] = numera * sqrt(Float32(N-1 )) / sqrt(var)
        p_T2s[t] = 1 - cdf(Normal(0, 1), T2_TVALS[t])
    end

    obj.Tstats = T2_TVALS
    obj.pvals = p_T2s

    return 
end

function get_h_vec_kernel!(x, y, N,  ϵ, w, w_sum, Cy, Cxy, Cyz, Cxyz, h, threshold)
    # Get the block's i value (outer loop)
    i = blockIdx().x + 1
    
    # Early return if block index is out of bounds
    if i > N || i < 2
        return nothing
    end
    
    # Thread index for inner loop parallelization
    tid = threadIdx().x
    nthreads = blockDim().x
    
    # Constants
    mu = Float32((2.0 * ϵ)^4)
    inv_w_sum = 1.0f0 / w_sum
    inv_mu = 2.0f0 / mu
    
    # Shared memory for partial sums
    shared_mem = @cuDynamicSharedMem(Float32, 4*nthreads)
    s_Cy = view(shared_mem, 1:nthreads)
    s_Cxy = view(shared_mem, (nthreads+1):(2*nthreads))
    s_Cyz = view(shared_mem, (2*nthreads+1):(3*nthreads))
    s_Cxyz = view(shared_mem, (3*nthreads+1):(4*nthreads))

    
    # Initialize thread local accumulators
    Cy_local = zero(Float32)
    Cxy_local = zero(Float32)
    Cyz_local = zero(Float32)
    Cxyz_local = zero(Float32)

    
    # Each thread processes its portion of j values
    @inbounds for j = (2+tid-1):nthreads:N
        if j != i

            w_j = w[j-1]      
            if  w_j > threshold
      
                # Calculate distances
                disx = abs(x[i-1] - x[j-1])
                disy = abs(y[i-1] - y[j-1])
                
                if disy <= ϵ
                    Cy_local += w_j
                    
                    if disx <= ϵ
                        Cxy_local += w_j
                    end
                    
                    disz = max(abs(y[i] - y[j]), disy)
                    if disz <= ϵ
                        Cyz_local += w_j
                        if disx <= ϵ
                            Cxyz_local += w_j
                        end
                    end
                end
            end
        end
    end
    
    # Store local results in shared memory
    s_Cy[tid] = Cy_local
    s_Cxy[tid] = Cxy_local
    s_Cyz[tid] = Cyz_local
    s_Cxyz[tid] = Cxyz_local

    
    sync_threads()
    

    # perform a parallel reduction
    d = Int32(1)                                                    # Int32
    while d < nthreads
        sync_threads()
        index = Int32(2) * d * (tid-Int32(1)) + Int32(1)         # Int32
        @inbounds if index <= nthreads
            if index + d <= nthreads
                s_Cy[index] += s_Cy[index+d]
                s_Cxy[index] += s_Cxy[index+d]
                s_Cyz[index] += s_Cyz[index+d]
                s_Cxyz[index] += s_Cxyz[index+d]
               
            end
        end
        d *= Int32(2)                                               # Int32
    end

    
  
    # Final calculation (only first thread)
    if tid == 1
 
        Cy_val = s_Cy[1]*inv_w_sum
        Cxy_val = s_Cxy[1]*inv_w_sum
        Cyz_val = s_Cyz[1]*inv_w_sum
        Cxyz_val = s_Cxyz[1]*inv_w_sum
       
        
        Cy[i] = Cy_val
        Cxy[i] = Cxy_val
        Cyz[i] = Cyz_val
        Cxyz[i] = Cxyz_val

        h[i] = inv_mu * ((Cxyz_val * Cy_val - Cxy_val * Cyz_val) / 6.0f0)
    end
    
    sync_threads()

    return nothing
end

function get_h_vec_cross_terms_kernel!(x, y, N,  ϵ, w, w_sum, Cy, Cxy, Cyz, Cxyz, h, threshold)
    # Get the block's i value (outer loop)
    i = blockIdx().x + 1
    
    # Early return if block index is out of bounds
    if i > N || i < 2
        return nothing
    end
    
    # Thread index for inner loop parallelization
    tid = threadIdx().x
    nthreads = blockDim().x
    
    # Constants
    mu = Float32((2.0 * ϵ)^4)
    inv_w_sum = 1.0f0 / w_sum
    inv_mu = 2.0f0 / mu
    
    # Shared memory for partial sums
    shared_mem = @cuDynamicSharedMem(Float32, nthreads)
    s_htemp = view(shared_mem, (1):(nthreads))

    htemp_local = zero(Float32)
    @inbounds for j = (2+tid-1):nthreads:N
        if j != i

            w_j = w[j-1]
                
            if  w_j > threshold
               
                
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
                htemp_local += inv_mu * (Cxyz[j] * IYij + IXYZij * Cy[j] - 
                Cxy[j] * IYZij - IXYij * Cyz[j]) / (6.0f0 * w_sum)
            end
        end
    end

    s_htemp[tid] = htemp_local
    
    sync_threads()
    
    # perform a parallel reduction
    d = Int32(1)                                                    # Int32
    while d < nthreads
        sync_threads()
        index = Int32(2) * d * (tid-Int32(1)) + Int32(1)         # Int32
        @inbounds if index <= nthreads
            if index + d <= nthreads
                s_htemp[index] += s_htemp[index+d]
            end
        end
        d *= Int32(2)                                               # Int32
    end

    # Final calculation (only first thread)
    if tid == 1
        h[i] += s_htemp[1]
    end

    return nothing
end


function compute_h_weighted_kernel!(h_weighted, h, weight_reduction, N)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if idx <= N-1
        @inbounds h_weighted[idx] = h[idx+1] * weight_reduction[idx]
    end
    
    return nothing
end

function adjust_h_vec_kernel!(h_vec_adjusted, h_weighted, numera, N)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if idx <= N
        @inbounds h_vec_adjusted[idx] = h_weighted[idx] - numera
    end
    
    return nothing
end

function HAC_variance_cuda(h, N, m)
    K = floor(Int, sqrt(sqrt(N)))
    
    # Create kernel weights
    ohm = CuArray(Float32[1.0f0; 2.0f0 * (1.0f0 .- (1:K-1) / K)])
    cov = CUDA.zeros(Float32, K)
    
    # Launch kernel for covariance calculation
    @cuda threads=128 blocks=ceil(Int, K/128) HAC_covariance_kernel!(
        cov, h, N, m, K
    )

    CUDA.synchronize()
    println(cov)
    
    # Compute final variance
    VT2 = 9.0f0 * CUDA.dot(ohm, cov)
    return VT2
end

function HAC_covariance_kernel!(cov, h, N, m, K)
    k = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if k <= K
        sum_temp = 0.0f0
        for i in (m+k):N
            sum_temp += h[i] * h[i-k+1]
        end
        cov[k] = sum_temp / (N - m - k + 1)
    end
    
    return nothing
end