mutable struct NPTVGC_test
    x::Vector{Float64}
    y::Vector{Float64}
    ssize::Int
    lags::Int
    dates::Vector{Int}
    offset1::Int
    offset2::Int
    weights::Union{Nothing, String}
    ϵ::Union{Nothing,Float64, Vector{Float64}}
    a_ϵ::Union{Nothing, Float64}
    b_ϵ::Union{Nothing, Float64}
    γ::Union{Nothing, Float64}
    a_γ::Union{Nothing, Float64}
    b_γ::Union{Nothing, Float64}
    max_iter::Int
    max_iter_outer::Int
    g_tol::Float64
    show_trace::Bool
    show_warnings::Bool
    silent::Bool
    filter::Union{Nothing, String}
    Tstats::Union{Nothing, Vector{Float64}}
    pvals::Vector{Float64}
    cores::Int

    function NPTVGC_test(x::Vector{Float64}, y::Vector{Float64})
        # H0: y does not Granger-cause x
        ssize = length(x)
        lags = 1  # Assuming x and y are dimensions
        dates = collect(1:ssize)
        offset1 = min(50, round(ssize / 10))
        offset2 = 2 * offset1
        weights = "e"
        filter = "smoothing"
        pvals = Vector{Float64}[]  # Initialize as an empty array of Float64 arrays
        γ = 0.90
        a_γ = 0.85
        b_γ = 0.99999
        ϵ = 0.5
        a_ϵ = 0.001
        b_ϵ = 4
        # 100, 100, 0.000001, true, true, false,
        max_iter = 100
        max_iter_outer = 100
        g_tol = 0.0001
        show_trace = true
        show_warnings = true
        silent = false

        new(x, y, ssize, lags, dates, offset1, offset2, weights, ϵ, a_ϵ, b_ϵ, γ, a_γ, b_γ, max_iter, max_iter_outer, g_tol, show_trace, show_warnings, silent, filter, nothing, pvals, 1)
    end
end