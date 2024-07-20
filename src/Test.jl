mutable struct NPTVGC_test
    x::Vector{Float64}
    y::Vector{Float64}
    ssize::Int
    lags::Int
    dates::Vector{Int}
    offset1::Int
    offset2::Int
    weights::Union{Nothing, String}
    ϵ::Union{Nothing,Float64}
    γ::Union{Nothing, Float64}
    filter::Union{Nothing, String}
    Tstats::Union{Nothing, Vector{Float64}}
    pvals::Vector{Float64}
    cores::Int

    function NPTVGC_test(x::Vector{Float64}, y::Vector{Float64})
        ssize = length(x)
        lags = 1  # Assuming x and y are dimensions
        dates = collect(1:ssize)
        offset1 = min(50, round(ssize / 10))
        offset2 = 2 * offset1
        weights = "e"
        filter = "smoothing"
        pvals = Vector{Float64}[]  # Initialize as an empty array of Float64 arrays
        γ = 1.0
        ϵ = 0.5
        new(x, y, ssize, lags, dates, offset1, offset2, weights, ϵ, γ, filter, nothing, pvals, 1)
    end
end