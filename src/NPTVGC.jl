module NPTVGC

# Place all your imports here
using Distributions
using Random
using LinearAlgebra
using Distributed
using ARCHModels
using Printf
using Metal

include("Test.jl")
include("Weights.jl")
include("Preprocessing.jl")
include("TestStatistic.jl")

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



end # module NPTVGC
