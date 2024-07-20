mutable struct NPTVGC_test
    x::Vector{Float64}
    y::Vector{Float64}
    ssize::Int
    dim::Int
    dates::Vector{Int}
    offset1::Int
    offset2::Int
    weights::Union{Nothing, String}
    ϵ::Union{Nothing, Vector{Float64}}
    filter::Union{Nothing, String}
    Tstat::Union{Nothing, Vector{Float64}}
    pvals::Vector{Vector{Float64}}
    cores::Int

    function NPTVGC_test(x::Vector{Float64}, y::Vector{Float64})
        ssize = length(x)
        dim = 2  # Assuming x and y are dimensions
        dates = collect(1:ssize)
        offset1 = min(50, round(ssize / 10))
        offset2 = 2 * offset1
        weights = "e"
        filter = "smoothing"
        pvals = Vector{Float64}[]  # Initialize as an empty array of Float64 arrays

        new(x, y, ssize, dim, dates, offset1, offset2, weights, nothing, filter, nothing, pvals, 1)
    end
end