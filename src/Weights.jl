
function weights!(time::Tuple{Int, Int}, g::Float64, type_::String, filter_::String)
    """
    Calculate weights for filtering or smoothing operations.

    Parameters:
    - time : Tuple of Int
        Contains two elements, (s, T)
    - g : Float64
        Smoothing parameter
    - type_ : String
        'e' for exponential, 'q' for quadratic
    - filter_ : String
        'filtering', 'smoothing', or 'CVsmo'

    Returns:
    Array of Float64
        Array of weights, shape depends on the filter type
    """
    if filter_ == "filtering"
        if g == 1
            w = ones(Float64, time[1]) ./ time[1]
        else
            t = 1:time[1]
            if type_ == "e"
                w = ((1-g) / (1-g^time[1])) * g.^(time[1] .- t)
            elseif type_ == "q"
                w0 = (1 - (1-g)^2 .* (t .- time[1]).^2) .* (t .>= max(1, ceil(time[1] - 1 / (1-g))))
                w = w0 / sum(w0)
            else
                error("type \"$(type_)\" not allowed.")
            end
        end
    elseif filter_ == "smoothing"
        if g == 1
            w = ones(Float64, time[2]) / time[2]
        else
            t = 1:time[2]
            if type_ == "e"
                w = ((1-g) / (1+g-g^time[1]-g^(time[2]-time[1]+1))) * g.^abs.(time[1] .- t)
            elseif type_ == "q"
                w0 = (1 - (1-g)^2 .* (t .- time[1]).^2) .* ((t .>= max(1, ceil(time[1] - 1/(1-g)))) .& (t .<= min(time[2], floor(time[1] + 1/(1-g)))))
                w = w0 / sum(w0)
            else
                error("type \"$(type_)\" not allowed.")
            end
        end
    elseif filter_ == "CVsmo"
        if g == 1
            w = ones(Float64, time[2]) / (time[2] - 1)
            w[time[1]] = 0.0
        else
            t = 1:time[2]
            if type_ == "e"
                w0 = ((1-g) / (2*g-g^time[1]-g^(time[2]-time[1]+1))) * g.^abs.(time[1] .- t)
                w0[time[1]] = 0.0
                w = w0 / sum(w0)
            elseif type_ == "q"
                w0 = (1 - (1-g)^2 .* (t .- time[1]).^2) .* ((t .>= max(1, ceil(time[1] - 1/(1-g)))) .& (t .<= min(time[2], floor(time[1] + 1/(1-g)))))
                w0[time[1]] = 0.0
                w = w0 / sum(w0)
            else
                error("type \"$(type_)\" not allowed.")
            end
        end
    else
        error("filter \"$(filter_)\" not allowed")
    end

    return w
end


function single_cv_smooth_weight!(time::Tuple{Int, Int}, g::Float64, N::Int)
    """
    Calculate weights for filtering or smoothing operations.

    Parameters:
    - time : Tuple of Int
        Contains two elements, (s, t)
    - g : Float64
        Smoothing parameter
    - type_ : String
        'e' for exponential, 'q' for quadratic
    - filter_ : String
        'filtering', 'smoothing', or 'CVsmo'
    - i : Int
        Index of the weight to calculate

    Returns:
    Float64
        Weight at index i
    """
    
    w0 = ((1-g) / (2*g-g^time[1]-g^(N-time[1]+1))) * g^abs(time[1] - time[2])
    
    # if time[1] == time[2] w0 should equal 0
 
    return w0 
end