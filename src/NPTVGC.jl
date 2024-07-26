module NPTVGC

# Testing the null hypothesis that y does not Granger-cause x

# Place all your imports here
using Distributions
using Random
using LinearAlgebra
using Distributed
using ARCHModels
using Printf
using Optim

include("Test.jl")
include("Weights.jl")
include("Preprocessing.jl")
include("TestStatistic.jl")
include("CVLLK.jl")
include("Estimate.jl")
include("Utils.jl")


# Include all your functions and constants here
export 
# Structs 
       NPTVGC
# Weights functions
       weights!
# Preprocessing functions
       prefilter
       uniform
       normalise
# TestStatistic functions
        get_h_vec!
        max
        HAC_variance
        estimate_tv_tstats
# CVLLK functions
        lik_cv
        total_likelihoods!
# Estimate functions
        estimate_LDE
# Utils functions
        sigmoid_map
        inverse_sigmoid_map



end # module NPTVGC
