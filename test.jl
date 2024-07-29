using NPTVGC
using BenchmarkTools
x = randn(1001)
y = randn(1001)
y = 0.5 .* x .+ y
x = x[1:1000]
y = y[2:1001]
x = NPTVGC.normalise(x)
y = NPTVGC.normalise(y)
obj = NPTVGC.NPTVGC_test(x, y)  # Replace with the actual type of your object
obj.filter = "smoothing"
obj.max_iter = 1
obj.max_iter_outer = 1
# Set other properties of obj as needed
w = [NPTVGC.weights!((i, obj.ssize), 0.5, obj.weights, "CVsmo") for i in 1:obj.ssize]

function get_weights(gamma)
    for i in 1:obj.ssize
        w[i] = NPTVGC.weights!((i, 1000), gamma, "e", "CVsmo")
    end
    
    return 
end

function main1()

    benchmark = @btime  NPTVGC.lik_cv(obj, [0.990, 0.65765])
    println("Benchmark time: ", benchmark)
    #println("Negative likelihood: ", neg_lik)
    # benchmark = @btime NPTVGC.total_likelihoods!(obj.x, obj.y, obj.ssize, obj.lags, obj.lags, 0.5, 0.9)
    # println("Benchmark time: ", benchmark)

    # # Test that estimate_LDE does not throw an error
    # NPTVGC.estimate_LDE(obj)
    # println(obj.γ)
    # println(obj.ϵ)
    # # println("Benchmark time: ", benchmark)
end