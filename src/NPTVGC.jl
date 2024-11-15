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
using LoopVectorization
using CUDA

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
       single_cv_smooth_weight!
# Preprocessing functions
       prefilter
       uniform
       normalise
# TestStatistic functions
        #get_h_vec!
        get_h_matrix_weighted!
        max
        HAC_variance
        estimate_tv_tstats
        get_indicator_matrices!
# CVLLK functions
        lik_cv
        total_likelihoods!
        lik_cv_cuda
        launch_total_likelihoods_cuda!
        
# Estimate functions
        estimate_LDE
# Utils functions
        sigmoid_map
        inverse_sigmoid_map



end # module NPTVGC
