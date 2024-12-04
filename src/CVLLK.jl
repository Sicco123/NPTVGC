function bounded_log(x)
    return log(max(x, 1e-8))
end

function lik_cv(obj, pv)

    #weights_vec = [weights!((i, obj.ssize), γ, obj.weights, "CVsmo") for i in 1:obj.ssize]
    h_lik = total_likelihoods!(obj.x, obj.y, obj.ssize, pv...)
  
    L = sum(h_lik) #sum((h_lik[obj.offset1:end-obj.offset1]))
    neg_likelihood = -L / obj.ssize

    return neg_likelihood
end


function lik_cv_cuda(obj, pv)

    #weights_vec = [weights!((i, obj.ssize), γ, obj.weights, "CVsmo") for i in 1:obj.ssize]
    h_lik = launch_total_likelihoods_cuda!(obj.x, obj.y, obj.ssize, pv...)
  
    L = sum(h_lik) #sum((h_lik[obj.offset1:end-obj.offset1]))
    neg_likelihood = -L / obj.ssize

    return neg_likelihood
end


function total_likelihoods_kernel!(h, x, y, N, γ, ϵ, float_type, threshold)
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
    mu = 2.0 * ϵ
    mid = N - 1
    t = i - 1
    
    # Shared memory for partial sums
    shared_mem = @cuDynamicSharedMem(float_type, 5*nthreads)
    s_Cy = view(shared_mem, 1:nthreads)
    s_Cxy = view(shared_mem, (nthreads+1):(2*nthreads))
    s_Cyz = view(shared_mem, (2*nthreads+1):(3*nthreads))
    s_Cxyz = view(shared_mem, (3*nthreads+1):(4*nthreads))
    s_wsum = view(shared_mem, (4*nthreads+1):(5*nthreads))
    
    # Initialize thread local accumulators
    Cy_local = zero(float_type)
    Cxy_local = zero(float_type)
    Cyz_local = zero(float_type)
    Cxyz_local = zero(float_type)
    w_sum_local = zero(float_type)
    
    # Calculate part_1 (same for all threads in block)
    part_1 = (1.0 - γ) / (2.0*γ - γ^t - γ^(N-t+1))
    
    # Each thread processes its portion of j values
    @inbounds for j = (2+tid-1):nthreads:N
        if j != i
            # Calculate weight
            dist_from_mid = abs(mid - t + j - 1)
            
            weight_reduction = 1#γ^dist_from_mid
            
            if weight_reduction > threshold
                w_j = 1#weight_reduction * part_1 #γ^dist_from_mid
                
                w_sum_local += w_j
                
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
    s_wsum[tid] = w_sum_local
    

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
                s_wsum[index] += s_wsum[index+d]
            end
        end
        d *= Int32(2)                                               # Int32
    end

  
    # Final calculation (only first thread)
    if tid == 1
 
        Cy_val = s_Cy[1]
        Cxy_val = s_Cxy[1]
        Cyz_val = s_Cyz[1]
        Cxyz_val = s_Cxyz[1]
        w_sum = s_wsum[1]
        
        h_val = (bounded_log(Cxyz_val/mu^3/w_sum) + 
                bounded_log(Cy_val/mu/w_sum) + 
                bounded_log(Cxy_val/mu^2/w_sum) + 
                bounded_log(Cyz_val/mu^2/w_sum))
        
        h[i] = h_val
    end
    
    return nothing
end


function total_likelihoods_kernel!(h, x, y, N, γ, ϵ1, ϵ2, ϵ3,  float_type, threshold)
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
    mu1 = 2.0*ϵ1
    mu2 = 2.0*ϵ2
    mu3 = 2.0*ϵ3
    mid = N - 1
    t = i - 1
    
    # Shared memory for partial sums
    shared_mem = @cuDynamicSharedMem(float_type, 5*nthreads)
    s_Cy = view(shared_mem, 1:nthreads)
    s_Cxy = view(shared_mem, (nthreads+1):(2*nthreads))
    s_Cyz = view(shared_mem, (2*nthreads+1):(3*nthreads))
    s_Cxyz = view(shared_mem, (3*nthreads+1):(4*nthreads))
    s_wsum = view(shared_mem, (4*nthreads+1):(5*nthreads))
    
    # Initialize thread local accumulators
    Cy_local = zero(float_type)
    Cxy_local = zero(float_type)
    Cyz_local = zero(float_type)
    Cxyz_local = zero(float_type)
    w_sum_local = zero(float_type)
    
    # Calculate part_1 (same for all threads in block)
    part_1 = (1.0 - γ) / (2.0*γ - γ^t - γ^(N-t+1))

    
    # Each thread processes its portion of j values
    @inbounds for j = (2+tid-1):nthreads:N
        if j != i
            # Calculate weight
            dist_from_mid = abs(mid - t + j - 1)
            
            weight_reduction = γ^dist_from_mid
            
            if weight_reduction > threshold
                w_j =  weight_reduction * part_1 #γ^dist_from_mid
                
                w_sum_local += w_j
                
                disx = disy = 0.0
                disx = abs(x[i-1] - x[j-1])
                disy = abs(y[i-1] - y[j-1])

                # f(xyz)
                if disy <= ϵ1 
                
                    Cy[i] += w_j
                    disx_ϵ = disx <= ϵ2 
                    disx_ϵ && (Cxy[i] += w_j)
                    disz = max(abs(y[i] - y[j]), disy)
                    if disz <= ϵ3
                        Cyz[i] += w_j
                        disx_ϵ && (Cxyz[i] += w_j)
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
    s_wsum[tid] = w_sum_local
    

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
                s_wsum[index] += s_wsum[index+d]
            end
        end
        d *= Int32(2)                                               # Int32
    end

  
    # Final calculation (only first thread)
    if tid == 1
 
        Cy_val = s_Cy[1]
        Cxy_val = s_Cxy[1]
        Cyz_val = s_Cyz[1]
        Cxyz_val = s_Cxyz[1]
        w_sum = s_wsum[1]
        
        h[i] +=  (bounded_log(Cxyz_val/(mu1*mu2*mu3)/w_sum)+ bounded_log(Cy_val/mu1/w_sum) + bounded_log(Cxy_val/(mu1*mu2)/w_sum) + bounded_log(Cyz_val/(mu1*mu3)/w_sum)) 
        
        
    end
    
    return nothing
end

function launch_total_likelihoods_cuda!(x, y, N, γ, ϵ; float_type= Float64)


    # Configure dimensions - one block per i, many threads per block for j
    threads_per_block = 256  # Should be power of 2 for reduction
    blocks =  min(N - 1, 65535)  # One block per i value (excluding i=1)
    threshold = 1e-16  # Threshold for weight reduction
    # convert to float32
    x = convert(Array{float_type}, x)
    y = convert(Array{float_type}, y)
    γ = convert(float_type, γ)
    ϵ = convert(float_type, ϵ)

    # Allocate device arrays
    d_x = CuArray(x)
    d_y = CuArray(y)
    d_h = CUDA.zeros(float_type, N)
    
    # Calculate shared memory size
    shared_mem_size = 5 * threads_per_block * sizeof(float_type)  # Five arrays for reductions
    
    
    # Launch kernel
    @cuda blocks=blocks threads=threads_per_block shmem=shared_mem_size total_likelihoods_kernel!(
        d_h, d_x, d_y, N, γ, ϵ, float_type, threshold
    )
    
    # Copy results back to host
    h = Array(d_h)
    
    return h
end

function launch_total_likelihoods_cuda!(x, y, N, γ, ϵ1, ϵ2, ϵ3; float_type= Float32)
    # Configure dimensions - one block per i, many threads per block for j
    threads_per_block = 32  # Should be power of 2 for reduction
    blocks =  min(N - 1, 65535)  # One block per i value (excluding i=1)
    threshold = 1e-10  # Threshold for weight reduction
    # convert to float32
    x = convert(Array{float_type}, x)
    y = convert(Array{float_type}, y)
    γ = convert(float_type, γ)
    ϵ1 = convert(float_type, ϵ1)
    ϵ2 = convert(float_type, ϵ2)
    ϵ3 = convert(float_type, ϵ3)


    # Allocate device arrays
    d_x = CuArray(x)
    d_y = CuArray(y)
    d_h = CUDA.zeros(float_type, N)
    
    # Calculate shared memory size
    shared_mem_size = 5 * threads_per_block * sizeof(float_type)  # Five arrays for reductions
    
    
    # Launch kernel
    @cuda blocks=blocks threads=threads_per_block shmem=shared_mem_size total_likelihoods_kernel!(
        d_h, d_x, d_y, N, γ, ϵ1, ϵ2, ϵ3, float_type, threshold
    )
    
    # Copy results back to host
    h = Array(d_h)
    
    return h
end
function total_likelihoods!(x, y, N::Int,  γ, ϵ::Float64)
    
    mu = 2.0*ϵ
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
      
        h[i] += (bounded_log(Cxyz[i]/mu^3/w_sum)+ bounded_log(Cy[i]/mu/w_sum) + bounded_log(Cxy[i]/mu^2/w_sum) + bounded_log(Cyz[i]/mu^2/w_sum)) 
    
    end
    
      
    return h
end


function total_likelihoods!(x, y, N::Int,   γ,  ϵ1::Float64, ϵ2::Float64, ϵ3::Float64)
    
    mu1 = 2.0*ϵ1
    mu2 = 2.0*ϵ2
    mu3 = 2.0*ϵ3

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

                # f(xyz)
                if disy <= ϵ1 
                
                    Cy[i] += w_j
                    disx_ϵ = disx <= ϵ2 
                    disx_ϵ && (Cxy[i] += w_j)
                    disz = max(abs(y[i] - y[j]), disy)
                    if disz <= ϵ3
                        Cyz[i] += w_j
                        disx_ϵ && (Cxyz[i] += w_j)
                    end
                end

            end
        end
      
        h[i] += (bounded_log(Cxyz[i]/(mu1*mu2*mu3)/w_sum)+ bounded_log(Cy[i]/mu1/w_sum) + bounded_log(Cxy[i]/(mu1*mu2)/w_sum) + bounded_log(Cyz[i]/(mu1*mu3)/w_sum)) 
    
    end
    
      
    return h
end
