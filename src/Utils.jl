# Generalized sigmoidal mapping from the real line to an interval [a, b]
function sigmoid_map(x, a, b)
    range = b - a
    return a + range / (1 + exp(-x))
end

# Inverse mapping from an interval [a, b] back to the real line
function inverse_sigmoid_map(y, a, b)

    return log((y - a) / (b - y))
end