# Generalized sigmoidal mapping from the real line to an interval [a, b]
function sigmoid_map(x, a, b)
    range = b - a
    return a + range / (1 + exp(-x))
end

# Inverse mapping from an interval [a, b] back to the real line
function inverse_sigmoid_map(y, a, b)

    return log((y - a) / (b - y))
end

function A_mul_B!(C, A, B)
    @turbo for n ∈ indices((C,B), 2), m ∈ indices((C,A), 1)
        Cmn = zero(eltype(C))
        for k ∈ indices((A,B), (2,1))
            Cmn += A[m,k] * B[k,n]
        end
        C[m,n] = Cmn
    end
end