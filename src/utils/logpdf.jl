# Author: Ali Siahkoohi, alisk@gatech.edu
# Date: March 2022
# Copyright: Georgia Institute of Technology, 2022

export logpdf, gradlogpdf


function logpdf(μ::Float32, σ::Float32, X)

    f = -0.5f0 * ((X .- μ) / σ) .^ 2
    f = f .+ -5.0f-1 * log(2.0f0π) .- log(σ)

    return f
end

function gradlogpdf(μ::Float32, σ::Float32, X)

    g = -(X .- μ) / σ^2

    return g
end
