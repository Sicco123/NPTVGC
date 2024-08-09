using NPTVGC
using BenchmarkTools
using Distributions
using LinearAlgebra
using Random
using PrettyTables
function get_weights(gamma)
    for i in 1:obj.ssize
        w[i] = NPTVGC.weights!((i, 1000), gamma, "e", "CVsmo")
    end
    
    return 
end

function calculate_multivariate_log_likelihood(X)
    # transpose X 
    X = X'
    d, N = size(X)  # dimensions and number of data points
    mu = zeros(d)  # mean vector of the data
    sigma = Matrix{Float64}(I, d, d)  # covariance matrix of the data
    dist = MvNormal(mu, sigma)  # define the multivariate normal distribution
    log_likelihood = [pdf(dist, X[:, i]) for i in 1:N]  # calculate the log-likelihood
    return log_likelihood
end

function lik_cv(x, y, pv)


    #weights_vec = [weights!((i, obj.ssize), γ, obj.weights, "CVsmo") for i in 1:obj.ssize]
    h_lik = total_likelihoods!(x, y, size(y)[2],1, 1, pv...)

    L = sum(h_lik)#sum((h_lik[obj.offset1:end-obj.offset1]))
    neg_likelihood = -L / size(y)[2]

    return neg_likelihood
end




function total_likelihoods!(x, y, N::Int, m::Int, mmax::Int, ϵ::Float64, γ)
    
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
        w_sum = sum(w)

        #println(γ)
        #println("sum",w_sum)
        #println(calculate_multivariate_log_likelihood(y[1:1,1:end-1]))
        #println("w", w)
        #println(sum(w.*calculate_multivariate_log_likelihood(y[1:1,1:end-1])))

        h[i] +=  log(sum(w.*calculate_multivariate_log_likelihood(hcat(y[1,2:end],y[1,1:end-1],x[1,2:end])))/w_sum) + log(sum(w.*calculate_multivariate_log_likelihood(hcat(y[1,1:end-1])))/w_sum) + log(sum(w.*calculate_multivariate_log_likelihood(hcat(y[1,1:end-1],x[1,2:end])))/w_sum) + log(sum(w.*calculate_multivariate_log_likelihood(hcat(y[1,2:end],y[1,1:end-1])))/w_sum) #(Cxyz[i]/mu^3 + Cy[i]/mu + Cxy[i]/mu^2 + Cyz[i]/mu^2) /w_sum

    
    end

    return h
end


function main1()
    grid_size = 10
    # set seed
    Random.seed!(1234)
    N = 2000
    N_2 = 1000
    x = randn(1,N)
    y = randn(1,N+1)
    x[1:1,N_2:end] = y[1:1,N_2:end-1] #+ 1 * randn(1,250)
    y = y[1:1,2:end]
    #y[1,500:end] = y[1,500:end] + ones(size(y[1,500:end])).*20
    #y[1:1,2:1001] += 0.5 * x[1:1,1:1000] 
    # x = x[1:2000]
    # y = y[2:2001]
    #x = NPTVGC.normalise(x)
    #y = NPTVGC.normalise(y)
    test_obj = NPTVGC.NPTVGC_test(x[1, 1:end], y[1, 1:end])  # Replace with the actual type of your object
    test_obj.filter = "smoothing"
    test_obj.b_ϵ = 2
    test_obj.a_ϵ = 0.2
    # Set other properties of obj as needed

    test_obj = NPTVGC.NPTVGC_test( y[1, 1:end],x[1, 1:end])
    test_obj.γ = 0.99

    #NPTVGC.estimate_LDE_grid(test_obj, grid_size, grid_size)
    
    a = test_obj.x 
    b = test_obj.y 
    N = test_obj.ssize 
    ϵ = test_obj.ϵ 

    # b= NPTVGC.get_indicator_matrices!($a, $b, $N, $ϵ)
    # println(b)


    b =  NPTVGC.estimate_tv_tstats(test_obj, 1)
    println(b)
    p_vals_2 = test_obj.pvals
    tstats_2 = test_obj.Tstats

    # println(test_obj.γ)
    # println(test_obj.ϵ)
    println(p_vals_2[1:N_2])
    println()
    println()

    println(p_vals_2[N_2:end])
   
    # # println("Benchmark time: ", benchmark)
end

function benchmark_func(x, y, N, γ, ϵ)
    Ay, Axy, Ayz, Axyz = get_indicator_matrices!(x, y, N, ϵ)
    Cy, Cxy, Cyz, Cxyz = sum(Ay, dims = 2), sum(Axy, dims = 2), sum(Ayz, dims = 2), sum(Axyz, dims = 2)


    μ = (2.0 * ϵ)^(4)

    h = zeros(Float64, N, N)

    # Weight calcualtion 
    # vector N : 1 0 1 : N 
    weight_reduction = [abs(i - (N-2)) for i in 0:2*(N-2)]
    weight_reduction = γ.^weight_reduction

   

    mid = N -1
end


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