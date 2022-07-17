


function forward(X, Y, CH::ConditionalLayerHINT; logdet = nothing, x_lane = false)
    isnothing(logdet) ? logdet = (CH.logdet && ~CH.is_reversed) : logdet = logdet

    # Y-lane
    ~isnothing(CH.C_Y) ? (Yp = CH.C_Y.forward(Y)) : (Yp = copy(Y))
    logdet ? (Zy, logdet2) = CH.CL_Y.forward(Yp) : Zy = CH.CL_Y.forward(Yp)

    # X-lane: coupling layer
    ~isnothing(CH.C_X) ? (Xp = CH.C_X.forward(X)) : (Xp = copy(X))
    logdet ? (X, logdet1) = CH.CL_X.forward(Xp) : X = CH.CL_X.forward(Xp)

    # X-lane: conditional layer
    logdet ? (Zx, logdet3) = CH.CL_YX.forward(Yp, X)[2:3] : Zx = CH.CL_YX.forward(Yp, X)[2]

    if x_lane == false
        logdet ? (return Zx, Zy, logdet1 + logdet2 + logdet3) : (return Zx, Zy)
    else
        logdet ? (return Zx, Zy, logdet1 + logdet3) : (return Zx, Zy)
    end
end

function inverse(Zx, Zy, CH::ConditionalLayerHINT; logdet = nothing, x_lane = false)
    isnothing(logdet) ? logdet = (CH.logdet && CH.is_reversed) : logdet = logdet

    # Y-lane
    logdet ? (Yp, logdet1) = CH.CL_Y.inverse(Zy; logdet = true) :
    Yp = CH.CL_Y.inverse(Zy; logdet = false)
    ~isnothing(CH.C_Y) ? (Y = CH.C_Y.inverse(Yp)) : (Y = copy(Yp))

    # X-lane: conditional layer
    YZ = tensor_cat(Yp, Zx)
    logdet ? (X, logdet2) = CH.CL_YX.inverse(Yp, Zx; logdet = true)[2:3] :
    X = CH.CL_YX.inverse(Yp, Zx)[2]

    # X-lane: coupling layer
    logdet ? (Xp, logdet3) = CH.CL_X.inverse(X; logdet = true) :
    Xp = CH.CL_X.inverse(X; logdet = false)
    ~isnothing(CH.C_X) ? (X = CH.C_X.inverse(Xp)) : (X = copy(Xp))

    if x_lane == false
        logdet ? (return X, Y, logdet1 + logdet2 + logdet3) : (return X, Y)
    else
        logdet ? (return X, Y, logdet2 + logdet3) : (return X, Y)
    end
end


function inverse(Y, H::CouplingLayerHINT; scale = 1, permute = nothing, logdet = nothing)
    isnothing(logdet) ? logdet = (H.logdet && H.is_reversed) : logdet = logdet
    isnothing(permute) ? permute = H.permute : permute = permute

    # Permutation
    permute == "both" && (Y = H.C.forward(Y))
    Ya, Yb = tensor_split(Y)

    # Check for recursion
    recursive = false
    if typeof(Y) <: AbstractArray{Float32,4} && size(Y, 3) > 4
        recursive = true
    elseif typeof(Y) <: AbstractArray{Float32,5} && size(Y, 4) > 4
        recursive = true
    end

    # Coupling layer
    if recursive
        Xa, logdet1 = inverse(Ya, H; scale = scale + 1, permute = "none")
        if logdet
            Y_temp, logdet2 = H.CL[scale].inverse(Xa, Yb; logdet = true)[[2, 3]]
        else
            Y_temp = H.CL[scale].inverse(Xa, Yb)[2]
            logdet2 = 0.0f0
        end
        Xb, logdet3 = inverse(Y_temp, H; scale = scale + 1, permute = "none")
        logdet_full = logdet1 + logdet2 + logdet3
    else
        Xa = copy(Ya)
        if logdet
            Xb, logdet_full = H.CL[scale].inverse(Ya, Yb; logdet = true)[[2, 3]]
        else
            Xb = H.CL[scale].inverse(Ya, Yb)[2]
            logdet_full = 0.0f0
        end
    end

    # Initial permutation
    permute == "lower" && (Xb = H.C.inverse(Xb))
    X = tensor_cat(Xa, Xb)
    if permute == "full" || permute == "both"
        X = H.C.inverse(X)
    end

    if scale == 1
        logdet ? (return X, logdet_full) : (return X)
    else
        return X, logdet_full
    end
end


function ResidualBlock(
    n_in,
    n_hidden;
    k1 = 3,
    k2 = 3,
    p1 = 1,
    p2 = 1,
    s1 = 1,
    s2 = 1,
    fan = false,
    ndims = 2,
)

    k1 = Tuple(k1 for i = 1:ndims)
    k2 = Tuple(k2 for i = 1:ndims)
    # Initialize weights
    W1 = Parameter(glorot_uniform(k1..., n_in, n_hidden))
    W2 = Parameter(glorot_uniform(k2..., n_hidden, n_hidden))
    W3 = Parameter(1.0f-2 * glorot_uniform(k1..., 2 * n_in, n_hidden))
    b1 = Parameter(zeros(Float32, n_hidden))
    b2 = Parameter(zeros(Float32, n_hidden))

    return ResidualBlock(W1, W2, W3, b1, b2, fan, (s1, s2), (p1, p2))
end


function forward(X, Y, CH::NetworkConditionalHINT; logdet = nothing, x_lane = false)
    isnothing(logdet) ? logdet = (CH.logdet && ~CH.is_reversed) : logdet = logdet

    depth = length(CH.CL)
    logdet_ = 0.0f0
    logdet_x = 0.0f0
    for j = 1:depth
        logdet ? (X_, logdet1) = CH.AN_X[j].forward(X) : X_ = CH.AN_X[j].forward(X)
        logdet ? (Y_, logdet2) = CH.AN_Y[j].forward(Y) : Y_ = CH.AN_Y[j].forward(Y)
        logdet ? (X, Y, logdet3) = CH.CL[j].forward(X_, Y_; x_lane = x_lane) :
        (X, Y) = CH.CL[j].forward(X_, Y_)
        logdet && (logdet_ += (logdet1 + logdet2 + logdet3))
        logdet && (logdet_x += (logdet1 + logdet3))
    end
    if x_lane == false
        logdet ? (return X, Y, logdet_) : (return X, Y)
    else
        logdet ? (return X, Y, logdet_x) : (return X, Y)
    end
end

function inverse(Zx, Zy, CH::NetworkConditionalHINT; logdet = nothing, x_lane = false)
    isnothing(logdet) ? logdet = (CH.logdet && CH.is_reversed) : logdet = logdet

    depth = length(CH.CL)
    logdet_ = 0.0f0
    logdet_x = 0.0f0
    for j = depth:-1:1
        logdet ?
        (Zx_, Zy_, logdet1) = CH.CL[j].inverse(Zx, Zy; logdet = true, x_lane = x_lane) :
        (Zx_, Zy_) = CH.CL[j].inverse(Zx, Zy; logdet = false)
        logdet ? (Zy, logdet2) = CH.AN_Y[j].inverse(Zy_; logdet = true) :
        Zy = CH.AN_Y[j].inverse(Zy_; logdet = false)
        logdet ? (Zx, logdet3) = CH.AN_X[j].inverse(Zx_; logdet = true) :
        Zx = CH.AN_X[j].inverse(Zx_; logdet = false)
        logdet && (logdet_ += (logdet1 + logdet2 + logdet3))
        logdet && (logdet_x += (logdet1 + logdet3))
    end
    if x_lane == false
        logdet ? (return Zx, Zy, logdet_) : (return Zx, Zy)
    else
        logdet ? (return Zx, Zy, logdet_x) : (return Zx, Zy)
    end
end
