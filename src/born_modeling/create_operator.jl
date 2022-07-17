export create_operator, NoiseSampler, mix_data!

function create_operator(;
    nrec = 204,
    nsrc = 102,
    sigma = 2000.0f0,
    sim_src = false,
    src_enc = nothing,
)

    n = (256, 256)
    v0 = ones(Float32, n) * 2.5f0
    v0 = (v0 .* range(1.0f0, 4.5f0 / maximum(v0), length = n[1]))'
    s = 1.0f0 ./ v0
    s[:, 1:10] .= 1 / 1.5f0
    m0 = s .^ 2.0f0
    d = (20.0f0, 12.5f0)
    o = (0.0f0, 0.0f0)

    # Setup info and model structure.
    model0 = Model(n, d, o, m0)

    # Setup receiver geometry.
    xrec = range(0.0f0, stop = model0.d[1] * (model0.n[1] - 1.0f0), length = nrec)
    yrec = 0.0f0
    zrec = range(2.0f0 * model0.d[2], stop = 2.0f0 * model0.d[2], length = nrec)

    # Receiver sampling and recording time.
    timeR = 2000.0f0
    dtR = 1.378f0

    # Setup receiver structure.
    recGeometry = Geometry(xrec, yrec, zrec; dt = dtR, t = timeR, nsrc = nsrc)

    # Setup source geometry (cell array with source locations for each shot).
    if sim_src == false
        xsrc = convertToCell(
            range(0.0f0, stop = model0.d[1] * (model0.n[1] - 1.0f0), length = nsrc),
        )
        ysrc = convertToCell(range(0.0f0, stop = 0.0f0, length = nsrc))
        zsrc = convertToCell(
            range(2.0f0 * model0.d[2], stop = 2.0f0 * model0.d[2], length = nsrc),
        )
    else
        xsrc = Array{Any}(undef, nsrc)
        ysrc = Array{Any}(undef, nsrc)
        zsrc = Array{Any}(undef, nsrc)
        for j = 1:nsrc
            xsrc[j] = collect(
                range(0.0f0, stop = model0.d[1] * (model0.n[1] - 1.0f0), length = nsrc),
            )
            ysrc[j] = [0.0f0]
            zsrc[j] = collect(
                range(2.0f0 * model0.d[2], stop = 2.0f0 * model0.d[2], length = nsrc),
            )
        end
        xsrc = convert(Array{Array{Float32,1},1}, xsrc)
        ysrc = convert(Array{Array{Float32,1},1}, ysrc)
        zsrc = convert(Array{Array{Float32,1},1}, zsrc)
    end

    # Source sampling and number of time steps.
    timeS = 2000.0f0
    dtS = 1.378f0

    # Setup source structure.
    srcGeometry = Geometry(xsrc, ysrc, zsrc; dt = dtS, t = timeS)

    # Setup wavelet.
    f0 = 0.03f0
    if sim_src == false
        wavelet = ricker_wavelet(timeS, dtS, f0)
        q = judiVector(srcGeometry, wavelet)
    else
        if src_enc == nothing
            # Create array with different time shifts of the wavelet.
            src_enc = randn(Float32, nsrc, nsrc) ./ sqrt(nsrc)
        end
        wavelet = zeros(Float32, srcGeometry.nt[1], nsrc)
        for j = 1:nsrc
            wavelet[:, j] = ricker_wavelet(timeS, dtS, f0)
        end

        q = judiVector(srcGeometry, wavelet)
        for j = 1:nsrc
            q.data[j] .= q.data[j] .* src_enc[j:j, :]
        end
    end

    # Setup options for forward modeling.
    ntComp = get_computational_nt(srcGeometry, recGeometry, model0, dt = dtS)
    info = Info(prod(n), nsrc, ntComp)
    opt = Options(
        optimal_checkpointing = false,
        isic = false,
        space_order = 16,
        free_surface = false,
        save_data_to_disk = false,
        return_array = true,
    )

    # Setup operators.
    Pr = judiProjection(info, recGeometry)
    F0 = judiModeling(info, model0; options = opt)
    Ps = judiProjection(info, srcGeometry)
    Mr = judiTopmute(n, 10, 2)
    J = judiJacobian(Pr * F0 * adjoint(Ps), q)

    noise_dist =
        NoiseSampler(ricker_wavelet(timeS, dtS, f0)[1:50], sigma, Int(size(J)[1] / nsrc))

    if sim_src == false
        return (J, Mr), noise_dist
    else
        return (J, Mr), noise_dist, src_enc
    end
end


# Band-limited noise sampler.
mutable struct NoiseSampler
    op::Conv
    kernel::Array{Float32,1}
    sigma::Float32
    n::Int64
end

function NoiseSampler(kernel::Array{Float32,1}, sigma::Float32, n::Int64)
    op = Conv(reshape(kernel, :, 1, 1), zeros(Float32, 1), identity; pad = SamePad())
    return NoiseSampler(op, kernel, sigma, n)
end

function rand(NS::NoiseSampler)
    e = randn(Float32, (NS.n, 1, 1))
    e = NS.op(e)
    e *= 1.0f0 / norm(e)
    e *= sqrt(NS.n) * NS.sigma
    return e[:, 1, 1]
end

# Mix data according to the source encodings.
function mix_data!(d, src_enc)
    @info "Mixing shots to create simultaneous source data"
    @assert size(src_enc, 2) == size(d, 1)
    d_tmp = zeros(Float32, size(d))
    p = Progress(size(d, 1))
    for i = 1:size(d, 1)
        Base.flush(Base.stdout)
        for j = 1:size(d, 1)
            d_tmp[i, :] = d_tmp[i, :] + d[j, :] * src_enc[i, j]
        end
        ProgressMeter.next!(p; showvalues = [(:Shot, i)])
    end
    d .= d_tmp
    d_tmp = 0.0f0
end
