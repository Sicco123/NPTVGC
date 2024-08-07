using NPTVGC
#using BenchmarkTools
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
    x = randn(1,5000)
    y = randn(1,5000)
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

    println(lik_cv(x, y, [0.5, 0.99999]))
    

    _, _, neg_lik, results = NPTVGC.estimate_LDE_grid(test_obj, grid_size, grid_size)
    println("Negative likelihood: ", neg_lik)
    # print likelihoods in matrix form
    matrix = reshape([x[3] for x in results], grid_size, grid_size)
    pretty_table(matrix)

    # println(test_obj.γ)
    # println(test_obj.ϵ)
    #neg_lik = NPTVGC.lik_cv(test_obj, [ 0.9999, 0.8])
    #println(neg_lik)
    #println("Negative likelihood: ", neg_lik)
    # benchmark = @btime NPTVGC.total_likelihoods!(obj.x, obj.y, obj.ssize, obj.lags, obj.lags, 0.5, 0.9)
    # println("Benchmark time: ", benchmark)

    # Test that estimate_LDE does not throw an error

   

    # print log likelihood of y 
    #pdf_vals = calculate_multivariate_log_likelihood(hcat(y[1,1:end-1]))#,x[1,2:end]))
    # h_lik = total_likelihoods!(x, y, test_obj.ssize, test_obj.lags, test_obj.lags, test_obj.ϵ, test_obj.γ)
    # L = sum(log.(h_lik[2:end]))
    # neg_likelihood = -L / test_obj.ssize
    
    # println("Log likelihood of y: ", neg_likelihood)

    NPTVGC.estimate_tv_tstats(test_obj, 1)
    p_vals_1 = test_obj.pvals 
    tstats_1 = test_obj.Tstats

    println(test_obj.γ)
    println(test_obj.ϵ)
    #println(p_vals_1)
    
    test_obj = NPTVGC.NPTVGC_test( y[1, 1:end],x[1, 1:end])


    NPTVGC.estimate_LDE_grid(test_obj, grid_size, grid_size)
    
    NPTVGC.estimate_tv_tstats(test_obj, 1)
    p_vals_2 = test_obj.pvals
    tstats_2 = test_obj.Tstats

    println(test_obj.γ)
    println(test_obj.ϵ)
    #println(p_vals_2)
   
    # # println("Benchmark time: ", benchmark)
end