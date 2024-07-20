
function insertionSort!(X::Vector{Float64}, S::Vector{Int})
    M = length(X)
    I = collect(1:M)
    for i = 2:M
        R = X[i]
        r = i
        j = i - 1
        while j >= 1 && X[j] > R
            X[j+1] = X[j]
            I[j+1] = I[j]
            j -= 1
        end
        X[j+1] = R
        I[j+1] = r
    end
    for i = 1:M
        S[I[i]] = i
    end
end

function uniform(X::Vector{Float64})
    M = length(X)
    I = Vector{Int}(undef, M)
    insertionSort!(X, I)
    for i = 1:M
        X[i] = I[i] / M * 3.464101615  # to make unit variance
    end
end

function normalise(x::Vector{Float64})
    mean_x = mean(x)
    var_x = var(x, mean=mean_x)
    x .-= mean_x
    x ./= sqrt(var_x)
end

function prefilter(X, GARCH_model, ARMA_model)
    model = UnivariateARCHModel(GARCH_model, X; meanspec=ARMA_model)
    fit!(model)
    resids = residuals(model)
    return resids
end