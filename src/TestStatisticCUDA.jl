function estimate_tv_tstats_cuda_timed(obj)
     # Initialize timing dictionaries
     total_times = Dict{String, Float64}()
     per_iteration_times = Dict{String, Vector{Float64}}()
     
     x = obj.x
     y = obj.y
     N = length(x)
     ϵ = obj.ϵ
     γ = obj.γ
 
     # Time initialization
     init_time = @elapsed begin
         x_d = CuArray{Float32}(x)
         y_d = CuArray{Float32}(y)
         ϵ = Float32(ϵ)
         γ = Float32(γ)
         
         n_tests = N-1
         T2_TVALS = Vector{Float32}(undef, n_tests)
         p_T2s = Vector{Float32}(undef, n_tests)
         
         mid = N-1
         weight_reduction = Vector{Float32}(undef, 2*(N-2)+1)
         @inbounds for i in 0:2*(N-2)
             weight_reduction[i+1] = Float32(γ^abs(i - (N-2)))
         end
         weight_reduction = CuArray{Float32}(weight_reduction)
 
         Cy = CUDA.zeros(Float32, N)
         Cxy = CUDA.zeros(Float32, N)
         Cyz = CUDA.zeros(Float32, N)
         Cxyz = CUDA.zeros(Float32, N)
         h = CUDA.zeros(Float32, N)

         max_blocks = min(cld(N-1, 256), 1024)
         output_buffer = CUDA.zeros(Float32, max_blocks)
         
         h_weighted = CUDA.zeros(Float32, N-1)
         h_vec_adjusted = CUDA.zeros(Float32, N-1)
     end
     total_times["initialization"] = init_time
     
     # Initialize per-iteration timing vectors
     timing_categories = [
         "w_sum_calculation",
         "array_reset",
         "h_vec_kernel",
         "cross_terms_kernel",
         "weighted_h_calculation",
         "numera_reduction",
         "h_vec_adjustment",
         "variance_calculation",
         "final_stats"
     ]
     
     for category in timing_categories
         per_iteration_times[category] = Float64[]
     end
     
     # Main loop with timing
     total_loop_time = @elapsed begin
         for t in 1:n_tests
             # Time w_sum calculation
             t_w_sum = @elapsed begin
                 w_sum = Float32((1 + γ - γ^t - γ^(N-t)) / (1-γ))
                 w_start = mid-t+1
                 w_end = length(weight_reduction)-t+1
             end
             push!(per_iteration_times["w_sum_calculation"], t_w_sum)
             
             # Time array reset
             t_reset = @elapsed begin
                 CUDA.fill!(Cy, 0.0f0)
                 CUDA.fill!(Cxy, 0.0f0)
                 CUDA.fill!(Cyz, 0.0f0)
                 CUDA.fill!(Cxyz, 0.0f0)
                 CUDA.fill!(h, 0.0f0)
                 CUDA.fill!(output_buffer, 0.0f0)
             end
             push!(per_iteration_times["array_reset"], t_reset)
             
             # Time h_vec kernel
             t_h_vec = @elapsed begin
                 threads_per_block = 128
                 blocks = min(N - 1, 65535)
                 shared_mem_size = 4 * threads_per_block * sizeof(Float32)
                 
                 @cuda blocks=blocks threads=threads_per_block shmem=shared_mem_size get_h_vec_kernel!(
                     x_d, y_d, N, ϵ, view(weight_reduction, w_start:w_end),
                     w_sum, Cy, Cxy, Cyz, Cxyz, h, Float32(1e-6)
                 )
                 # synchronize 
                 CUDA.synchronize()
             end
             push!(per_iteration_times["h_vec_kernel"], t_h_vec)
             
             # Time cross terms kernel
             t_cross = @elapsed begin
                 shared_mem_size = threads_per_block * sizeof(Float32)
                 @cuda blocks=blocks threads=threads_per_block shmem=shared_mem_size get_h_vec_cross_terms_kernel!(
                     x_d, y_d, N, ϵ, view(weight_reduction, w_start:w_end),
                     w_sum, Cy, Cxy, Cyz, Cxyz, h, Float32(1e-6)
                 )
                 CUDA.synchronize()
             end
             push!(per_iteration_times["cross_terms_kernel"], t_cross)
             
             # Time weighted h calculation
             t_weighted = @elapsed begin
                 @cuda threads=128 blocks=ceil(Int, (N-1)/128) compute_h_weighted_kernel!(
                     h_weighted, h, view(weight_reduction, w_start:w_end), N
                 )
                 CUDA.synchronize()
             end
             push!(per_iteration_times["weighted_h_calculation"], t_weighted)
             
             # Time numera reduction
             t_numera = @elapsed begin
                numera = optimized_numera_calculation!(output_buffer, h_weighted, w_sum, N-1)
                CUDA.synchronize()
             end
             push!(per_iteration_times["numera_reduction"], t_numera)
             
             # Time h_vec adjustment
             t_adjust = @elapsed begin
                 @cuda threads=128 blocks=ceil(Int, (N-1)/128) adjust_h_vec_kernel!(
                     h_vec_adjusted, h_weighted, numera, N-1
                 )
                 CUDA.synchronize()
             end
             push!(per_iteration_times["h_vec_adjustment"], t_adjust)
             
             # Time variance calculation
             t_variance = @elapsed begin
                 h_vec_adjusted_cpu = Array(h_vec_adjusted)
                 var = HAC_variance_orig(h_vec_adjusted_cpu, N-1, 1)
                 CUDA.synchronize()    
            end
             push!(per_iteration_times["variance_calculation"], t_variance)
             
             # Time final statistics
             t_stats = @elapsed begin
                 T2_TVALS[t] = numera * sqrt(Float32(N-1)) / sqrt(var)
                 p_T2s[t] = 1 - cdf(Normal(0, 1), T2_TVALS[t])
             end
             push!(per_iteration_times["final_stats"], t_stats)
         end
     end
     total_times["main_loop"] = total_loop_time
     
     # Calculate and return timing statistics
     timing_stats = Dict{String, NamedTuple{(:mean, :std, :min, :max, :total), Tuple{Float64, Float64, Float64, Float64, Float64}}}()
     
     for (category, times) in per_iteration_times
         timing_stats[category] = (
             mean = mean(times),
             std = std(times),
             min = minimum(times),
             max = maximum(times),
             total = sum(times)
         )
     end
     
     return (
         total_times = total_times,
         per_operation = timing_stats
     )
end
function reduce_sum_kernel!(output, input, n)
    # Get thread and block IDs
    tid = threadIdx().x
    bid = blockIdx().x
    
    # Shared memory for block-level reduction
    shared_mem = @cuDynamicSharedMem(Float32, 256)  # Using fixed 256 threads
    
    # Initialize local sum
    local_sum = Float32(0)
    
    # Grid-stride loop to handle large arrays
    i = tid + (bid-1) * blockDim().x
    while i <= n
        local_sum += input[i]
        i += blockDim().x * gridDim().x
    end
    
    # Load into shared memory
    shared_mem[tid] = local_sum
    sync_threads()
    
    # Perform reduction in shared memory
    for s in 128:-1:1
        if tid <= s
            shared_mem[tid] += shared_mem[tid + s]
        end
        sync_threads()
    end
    
    # Write result for this block
    if tid == 1
        output[bid] = shared_mem[1]
    end
    
    return nothing
end

function optimized_numera_calculation!(output, h_weighted, w_sum, N)
    threads = 256
    blocks = min(cld(N, threads), 1024)
    shmem = threads * sizeof(Float32)
    
    # Launch kernel for first reduction
    @cuda threads=threads blocks=blocks shmem=shmem reduce_sum_kernel!(output, h_weighted, N)
    
    # Get final sum from output array
    result = Array(output)[1:blocks] |> sum
    
    return result / w_sum
end


function estimate_tv_windowed_tstats_cuda(obj)

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
    weight_reduction_cpu = Vector{Float32}(undef, 2*(N-2)+1)
    @inbounds for i in 0:2*(N-2)
        weight_reduction_cpu[i+1] = Float32(γ^abs(i - (N-2)))
    end
    weight_reduction_squares_cpu = weight_reduction_cpu.^2
    
    weight_reduction = CuArray{Float32}(weight_reduction_cpu)
    weight_reduction_squares = CuArray{Float32}(weight_reduction_squares_cpu)

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
 
    
    window_size = Int(round((1 + γ - γ^(N/2) - γ^(N-N/2)) / (1-γ)))
    window_size = min(window_size, N-1)
    # make even 
    if window_size % 2 != 0
        window_size += 1
    end
    half_window = Int(window_size/2)

    for t in 1+half_window:n_tests-half_window
        

        ### ESTIMATION of DENSITIES
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

        shared_mem_size = threads_per_block * sizeof(Float32)  # Five arrays for reductions
        
        # Add cross terms 
        @cuda blocks=blocks threads=threads_per_block shmem=shared_mem_size get_h_vec_cross_terms_kernel!(
        x_d, y_d, N, ϵ, view(weight_reduction, w_start:w_end),
        w_sum, Cy, Cxy, Cyz, Cxyz, h, weight_threshold
        )

        ### CALCULATION OF T-STATS

        window_start = t-half_window
        window_end = t+half_window-1


        # Calculate numera using reduction

        numera = CUDA.reduce(+,view(h[window_start:window_end]))/(2*half_window) 
    

        # Adjust h_vec in parallel
        # @cuda threads=128 blocks=ceil(Int, (N-1)/128) adjust_h_vec_kernel!(
        #     h_vec_adjusted, h_weighted, numera, N-1
        # )
        @cuda threads=128 blocks=ceil(Int, (N-1)/128) adjust_h_vec_kernel!(
            view(h_vec_adjusted[window_start:window_end ]), view(h[window_start:window_end ]), numera, window_size-1
        )
        

        # move h_vec_adjusted to CPU
        h_vec_adjusted_cpu = Array(h_vec_adjusted[window_start:window_end ])

        
   
        # Calculate variance using HAC
        var = HAC_variance_orig(h_vec_adjusted_cpu, window_size-1, 1) #HAC_variance_cuda(h_vec_adjusted, N-1, 1) # 

        # Calculate final statistics on CPU
        T2_TVALS[t] = numera * sqrt(window_size) / sqrt(var)
        p_T2s[t] = 1 - cdf(Normal(0, 1), T2_TVALS[t])
    end

    obj.Tstats = T2_TVALS
    obj.pvals = p_T2s

    return 
end


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
    if γ == 1.0
        n_tests = 1
    end
    T2_TVALS = Vector{Float32}(undef, n_tests)
    p_T2s = Vector{Float32}(undef, n_tests)
    
    # Pre-calculate weight reduction array once
    mid = N-1
    weight_reduction = Vector{Float32}(undef, 2*(N-2)+1)
    @inbounds for i in 0:2*(N-2)
        weight_reduction[i+1] = Float32(γ^abs(i - (N-2)))
    end
    weight_reduction_squares = weight_reduction.^2
    
    weight_reduction = CuArray{Float32}(weight_reduction)
    weight_reduction_squares = CuArray{Float32}(weight_reduction_squares)


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

        shared_mem_size = threads_per_block * sizeof(Float32)  # Five arrays for reductions
        
        # Add cross terms 
        @cuda blocks=blocks threads=threads_per_block shmem=shared_mem_size get_h_vec_cross_terms_kernel!(
        x_d, y_d, N, ϵ, view(weight_reduction, w_start:w_end),
        w_sum, Cy, Cxy, Cyz, Cxyz, h, weight_threshold
        )


        # Calculate weighted h values
        @cuda threads=128 blocks=ceil(Int, (N-1)/128) compute_h_weighted_kernel!(
            h_weighted, h, view(weight_reduction, w_start:w_end), N
        )


        # Calculate numera using reduction

        numera = CUDA.reduce(+,h_weighted)/(w_sum) 

    

        # Adjust h_vec in parallel
        # @cuda threads=128 blocks=ceil(Int, (N-1)/128) adjust_h_vec_kernel!(
        #     h_vec_adjusted, h_weighted, numera, N-1
        # )
        @cuda threads=128 blocks=ceil(Int, (N-1)/128) adjust_h_vec_kernel!(
            h_vec_adjusted, h, numera, N-1
        )
        

        # move h_vec_adjusted to CPU
        h_vec_adjusted_cpu = Array(h_vec_adjusted)

        
        # Calculate variance using HAC
        var = HAC_variance_weighted(h_vec_adjusted_cpu, N-1, 1, Array(view(weight_reduction, w_start:w_end)), w_sum) #HAC_variance_cuda(h_vec_adjusted, N-1, 1) # 

        # Calculate final statistics on CPU
        w_normal_2 = (1/w_sum)^2*CUDA.reduce(+,weight_reduction_squares[w_start:w_end])

        T2_TVALS[t] = numera / sqrt(w_normal_2*var)
        p_T2s[t] = 1 - cdf(Normal(0, 1), T2_TVALS[t])

   
    end

    if γ == 1.0
        for t in 2:n_tests
            T2_TVALS[t] = T2_TVALS[1]
            p_T2s[t] = p_T2s[1]
        end
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

function HAC_variance_weighted(h, N, m, w, w_sum)
    K = floor(Int, sqrt(sqrt(N)))

    ohm = Float32[1.0f0; 2.0f0 * (1.0f0 .- (1:K-1) / K)]
    cov = zeros(Float32, K)

    # Determine autocovariance of h[i]
    w_k_sum = 0.0f0
    @inbounds for k in 1:K
        cov[k] = 0.0f0
        w_k_sum += w[k]
        for i in (m+k):N
            cov[k] += w[i]*h[i] * h[i-k+1]  # +1 for Julia's 1-based indexing
        end
        cov[k] /= (w_sum - m - w_k_sum +1)  # +1 for correct denominator in 1-based indexing
    end

    VT2 = 9.0f0 * dot(ohm, cov)

    return VT2
end
