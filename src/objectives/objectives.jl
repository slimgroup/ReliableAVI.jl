# Author: Ali Siahkoohi, alisk@gatech.edu
# Date: March 2022
# Copyright: Georgia Institute of Technology, 2022

export loss_supervised, loss_unsupervised_finetuning

function loss_unsupervised_finetuning(
    Net::ActNorm,
    Net_amortized::ReverseNetwork,
    Y_obs::Array{Float32,1},
    Zx::AbstractArray{Float32,4},
    Zy_obs::AbstractArray{Float32,4},
    J::JUDI.judiJacobian{Float32,Float32},
    Mr,
    sigma::Float32,
    sigma_prior::Float32,
    AN_x_rev::InvertibleNetworks.Reverse,
    nsrc::Int,
    device;
    grad::Bool = true,
)
    @assert size(Zx, 4) == 1
    # Normalizing flow maps Guassian latent variables to amortized network latent space
    Zx_amortized, logdet = Net.forward(reshape(Zx, (1, 1, :, size(Zx, 4))))
    if CUDA.functional()
        CUDA.reclaim()
    end
    Zx_amortized = reshape(Zx_amortized, size(Zx))

    # Amortized network gives low-fidelity posterior samples
    X, Y = Net_amortized.forward(Zx_amortized, Zy_obs)
    if CUDA.functional()
        CUDA.reclaim()
    end

    X = AN_x_rev.forward(wavelet_unsqueeze(X |> cpu))
    # Zx_amortized = wavelet_unsqueeze(Zx_amortized |> cpu)
    shape = size(X)

    # Residual.
    residual = J * Mr * convert(Array{Float32,1}, vec(X[:, :, 1, 1]')) - Y_obs

    # Objective function: KL(q(z) || p(z|y))
    f =
        -nsrc * sum(logpdf(0.0f0, sigma, residual)) -
        sum(logpdf(0.0f0, sigma_prior, Zx_amortized)) - logdet

    if grad
        ΔX = reshape(
            reshape(
                -nsrc * adjoint(Mr) * adjoint(J) * gradlogpdf(0.0f0, sigma, residual),
                shape[1],
                shape[2],
            )',
            shape[1],
            shape[2],
            1,
            1,
        )
        ΔX = wavelet_squeeze(AN_x_rev.backward(ΔX, X)[1]) |> device
        X = wavelet_squeeze(AN_x_rev.inverse(X)) |> device
        # Zx_amortized = wavelet_squeeze(Zx_amortized) |> device

        ΔZx = Net_amortized.backward(ΔX, 0.0f0 * ΔX, X, Y)[1]
        ΔZx -= gradlogpdf(0.0f0, sigma_prior, Zx_amortized)

        ΔZx = reshape(ΔZx, (1, 1, :, size(Zx, 4)))
        Zx_amortized = reshape(Zx_amortized, (1, 1, :, size(Zx, 4)))
        Net.backward(ΔZx, Zx_amortized)
        if CUDA.functional()
            CUDA.reclaim()
        end
        clear_grad!(Net_amortized)
        GC.gc()
    end
    return f / size(X, 4), X
end


function loss_supervised(
    Net::NetworkConditionalHINT,
    X::AbstractArray{Float32,4},
    Y::AbstractArray{Float32,4};
    grad::Bool = true,
)

    Zx, Zy, logdet = Net.forward(X, Y)
    if CUDA.functional()
        CUDA.reclaim()
    end
    z_size = size(Zx)

    f = sum(logpdf(0.0f0, 1.0f0, Zx))
    f = f + sum(logpdf(0.0f0, 1.0f0, Zy))
    f = f + logdet * z_size[4]

    if grad
        ΔZx = -gradlogpdf(0.0f0, 1.0f0, Zx) / z_size[4]
        ΔZy = -gradlogpdf(0.0f0, 1.0f0, Zy) / z_size[4]

        ΔX, ΔY = Net.backward(ΔZx, ΔZy, Zx, Zy)[1:2]
        if CUDA.functional()
            CUDA.reclaim()
        end
        GC.gc()

        return -f / z_size[4], ΔX, ΔY
    else
        return -f / z_size[4]
    end
end
