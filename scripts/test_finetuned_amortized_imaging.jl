# Authors: Ali Siahkoohi, alisk@gatech.edu
# Date: May 2022


using DrWatson
@quickactivate :ReliableAVI

using InvertibleNetworks
using HDF5
using Random
using Statistics
using ProgressMeter
using PyPlot
using Seaborn
using LinearAlgebra
using Flux
using PyCall: @py_str
include(scriptsdir(joinpath("seismic-imaging", "create_OOD_shots_and_RTM.jl")))

# Configure device
device = gpu

font_prop, sfmt = sef_plot_configs()
args = read_config("test_finetuned_amortized_imaging.json")
args = parse_input_args(args)

args["parihaka_label"] = 1
args["nsrc"] = 25
args["nrec"] = 204
args["idx"] = 50
args["sigma"] = 5000.0f0
args["epoch"] = 5

py"""
from scipy.signal import hilbert
import numpy as np

def normalize_std(mu, sigma):
    analytic_mu = hilbert(mu, axis=1)
    return sigma*np.abs(analytic_mu)/(np.abs(analytic_mu)**2 + 5e-1), analytic_mu
"""
normalize_std(mu, sigma) = py"normalize_std"(mu, sigma)


max_epoch = args["max_epoch"]
lr = args["lr"]
sigma = args["sigma"]
sigma_prior = args["sigma_prior"]
idx = args["idx"]
parihaka_label = args["parihaka_label"]
lr_step = args["lr_step"]
nsrc = args["nsrc"]
nrec = args["nrec"]
amortized_model_config = args["amortized_model_config"]
sim_name = args["sim_name"]
if args["epoch"] == -1
    args["epoch"] = args["max_epoch"]
end
epoch = args["epoch"]

loaded_keys = load_experiment(args, ["Params", "fval", "mval", "fval_eval"])
Params, fval, mval, fval_eval = loaded_keys["Params"],
loaded_keys["fval"],
loaded_keys["mval"],
loaded_keys["fval_eval"]

# Loading the pretrained model.
amortized_model_args = read_config(amortized_model_config)
loaded_keys = load_experiment(
    amortized_model_args,
    ["Params", "AN_params_x", "AN_params_y", "n_hidden", "depth"],
)
Params_amortized, AN_params_x, AN_params_y, n_hidden_amortized, depth_amortized =
    loaded_keys["Params"],
    loaded_keys["AN_params_x"],
    loaded_keys["AN_params_y"],
    loaded_keys["n_hidden"],
    loaded_keys["depth"]

AN_x = ActNorm(1)
AN_y = ActNorm(1)
put_params!(AN_x, convert(Array{Any,1}, AN_params_x))
put_params!(AN_y, convert(Array{Any,1}, AN_params_y))

# Create shot data and RTM image given a index for out-of-distribution seismic image.
X_fixed, _, Y_fixed = create_OOD_shots_and_RTM(args)
Y_fixed[1:10, :, :, :] .= 0.0f0

X_fixed = AN_x.forward(X_fixed)
Y_fixed = AN_y.forward(Y_fixed)
X_fixed = wavelet_squeeze(X_fixed)
Y_fixed = wavelet_squeeze(Y_fixed)
nx, ny, n_in = size(X_fixed)[1:3]

# Define network mapping its latent space to the pretrained model's latent space.
H = ActNorm(nx * ny * n_in, logdet = true)
put_params!(H, convert(Array{Any,1}, Params))

# Define the amortized network
CH = NetworkConditionalHINT(n_in, n_hidden_amortized, depth_amortized, logdet = false)
put_params!(CH, convert(Array{Any,1}, Params_amortized))

# Corresponding data latent variable
Zy_obs = CH.forward_Y(Y_fixed)

H = H |> device
CH = CH |> device
CHrev = reverse(CH)

# Testing data
test_size = 1000
Zx_test = randn(Float32, 1, 1, nx * ny * n_in, test_size)

# Draw low-fidelity posterior samples.
X_post_amortized = zeros(Float32, nx, ny, n_in, test_size)

test_batchsize = 4
test_loader = Flux.DataLoader(
    (Zx_test, repeat(Zy_obs, 1, 1, 1, test_size)),
    batchsize = test_batchsize,
    shuffle = false,
)

@info "Drawing low-fidelity posterior samples"
p = Progress(length(test_loader))
for (itr, (Zx_, Zy_)) in enumerate(test_loader)
    Base.flush(Base.stdout)
    counter = (itr - 1) * test_batchsize + 1
    Zx_ = reshape(Zx_, (nx, ny, n_in, size(Zx_, 4)))
    Zx_ = Zx_ |> device
    Zy_ = Zy_ |> device
    X_post_amortized[:, :, :, counter:(counter+size(Zx_)[4]-1)] =
        (CHrev.forward(Zx_, Zy_)[1] |> cpu)
    ProgressMeter.next!(p)
end


# Draw high-fidelity posterior samples.
X_post = zeros(Float32, nx, ny, n_in, test_size)

@info "Drawing multi-fidelity posterior samples"
p = Progress(length(test_loader))
for (itr, (Zx_, Zy_)) in enumerate(test_loader)
    Base.flush(Base.stdout)
    counter = (itr - 1) * test_batchsize + 1
    Zx_ = Zx_ |> device
    Zy_ = Zy_ |> device
    Zx_finetuned = H.forward(Zx_)[1]
    Zx_finetuned = reshape(Zx_finetuned, (nx, ny, n_in, size(Zx_, 4)))
    X_post[:, :, :, counter:(counter+size(Zx_)[4]-1)] =
        (CHrev.forward(Zx_finetuned, Zy_)[1] |> cpu)
    ProgressMeter.next!(p)
end

# Draw low-fidelity posterior samples.
X_prior = zeros(Float32, nx, ny, n_in, test_size)

test_batchsize = 4
test_loader = Flux.DataLoader(
    (Zx_test, randn(Float32, nx, ny, n_in, test_size)),
    batchsize = test_batchsize,
    shuffle = false,
)

@info "Drawing prior samples"
p = Progress(length(test_loader))
for (itr, (Zx_, Zy_)) in enumerate(test_loader)
    Base.flush(Base.stdout)
    counter = (itr - 1) * test_batchsize + 1
    Zx_ = reshape(Zx_, (nx, ny, n_in, size(Zx_, 4)))
    Zy_ = reshape(Zy_, (nx, ny, n_in, size(Zy_, 4)))
    Zx_ = Zx_ |> device
    Zy_ = Zy_ |> device
    X_prior[:, :, :, counter:(counter+size(Zx_)[4]-1)] = (CHrev.forward(Zx_, Zy_)[1] |> cpu)
    ProgressMeter.next!(p)
end

X_prior = wavelet_unsqueeze(X_prior)
X_prior = AN_x.inverse(X_prior)
X_prior[1:10, :, :, :] .= 0.0f0

X_post_amortized = wavelet_unsqueeze(X_post_amortized)
X_post_amortized = AN_x.inverse(X_post_amortized)
X_post_amortized[1:10, :, :, :] .= 0.0f0
X_post = wavelet_unsqueeze(X_post)
X_post = AN_x.inverse(X_post)
X_post[1:10, :, :, :] .= 0.0f0

# Some stats
# X_post_amortized_mean = mean(X_post_amortized; dims = 4)
X_post_amortized_std = std(X_post_amortized; dims = 4)
# X_post_mean = mean(X_post; dims = 4)
X_post_std = std(X_post; dims = 4)

X_post_cum_amortized_mean =
    cumsum(X_post_amortized, dims = 4) ./
    reshape(collect(1:size(X_post_amortized)[4]), (1, 1, 1, size(X_post_amortized)[4]))
X_post_amortized_mean = X_post_cum_amortized_mean[:, :, :, end:end]

X_post_cum_mean =
    cumsum(X_post, dims = 4) ./
    reshape(collect(1:size(X_post)[4]), (1, 1, 1, size(X_post)[4]))
X_post_mean = X_post_cum_mean[:, :, :, end:end]

X_fixed = wavelet_unsqueeze(X_fixed)
Y_fixed = wavelet_unsqueeze(Y_fixed)
Zy_obs = wavelet_unsqueeze(Zy_obs)

X_fixed = AN_x.inverse(X_fixed)
Y_fixed = AN_y.inverse(Y_fixed)
Y_fixed[1:10, :, :, :] .= 0.0f0

save_dict =
    @strdict max_epoch lr lr_step sigma sigma_prior epoch idx nsrc nrec parihaka_label amortized_model_config
save_path = plotsdir(sim_name, savename(save_dict; digits = 6))

save_dict =
    @strdict max_epoch lr lr_step sigma sigma_prior epoch idx nsrc nrec parihaka_label amortized_model_config X_fixed Y_fixed X_post_mean X_post_amortized_mean X_post_std X_post_amortized_std X_post_amortized X_post X_prior
wsave(joinpath(save_path, savename(save_dict, "jld2"; digits = 6)), save_dict)

spacing = [20.0, 12.5]
extent = [0.0, size(X_fixed, 1) * spacing[1], size(X_fixed, 2) * spacing[2], 0.0] / 1e3

signal_to_noise(xhat, x) = -20.0 * log(norm(x - xhat) / norm(x)) / log(10.0)

snr_list_amortized = []
snr_list_cum_amortized_mean = []
snr_list = []
snr_list_cum_mean = []
for j = 1:test_size
    push!(
        snr_list_amortized,
        signal_to_noise(X_post_amortized[:, :, :, j], X_fixed[:, :, :, 1]),
    )
    push!(
        snr_list_cum_amortized_mean,
        signal_to_noise(X_post_cum_amortized_mean[:, :, :, j], X_fixed[:, :, :, 1]),
    )
    push!(snr_list, signal_to_noise(X_post[:, :, :, j], X_fixed[:, :, :, 1]))
    push!(
        snr_list_cum_mean,
        signal_to_noise(X_post_cum_mean[:, :, :, j], X_fixed[:, :, :, 1]),
    )
end

X_post_amortized_mean_snr =
    signal_to_noise(X_post_amortized_mean[:, :, :, 1], X_fixed[:, :, :, 1])
X_post_mean_snr = signal_to_noise(X_post_mean[:, :, :, 1], X_fixed[:, :, :, 1])
Y_fixed_snr = signal_to_noise(Y_fixed[:, :, :, 1] / nrec, X_fixed[:, :, :, 1])


# Training loss
fig = figure("training logs", figsize = (7, 4))
if epoch == args["max_epoch"]
    plot(
        range(0, epoch, length = length(fval)),
        fval,
        color = "#4a4a4a",
        label = "Training",
    )
    plot(
        range(0, epoch, length = length(fval_eval)),
        fval_eval,
        color = "#a1a1a1",
        label = "Validation",
    )
else
    plot(
        range(0, epoch, length = length(fval[1:findfirst(fval .== 0.0f0)-1])),
        fval[1:findfirst(fval .== 0.0f0)-1],
        color = "#4a4a4a",
        label = "Training",
    )
    plot(
        range(0, epoch, length = length(fval_eval[1:findfirst(fval_eval .== 0.0f0)-1])),
        fval_eval[1:findfirst(fval_eval .== 0.0f0)-1],
        color = "#a1a1a1",
        label = "Validation",
    )
end
ticklabel_format(axis = "y", style = "sci", useMathText = true)
ylabel("Objective value")
title("Latent distribution correction")
xlabel("Number of passes through shots")
legend(loc = "upper right", ncol = 1, fontsize = 9)
wsave(joinpath(save_path, "log.png"), fig)
close(fig)



# Training loss
fig = figure("training logs", figsize = (7, 4))
if epoch == args["max_epoch"]
    fval_per_epoch =
        mean(reshape(fval, (Int(length(fval) / length(fval_eval))), :), dims = 1)
    plot(
        range(0, epoch, length = length(fval_per_epoch)),
        vec(fval_per_epoch),
        color = "#4a4a4a",
    )
else
    fval_per_epoch = fval[1:findfirst(fval .== 0.0f0)-1]
    fval_per_epoch =
        mean(reshape(fval_per_epoch, (Int(length(fval) / length(fval_eval))), :), dims = 1)
    plot(
        range(0, epoch, length = length(fval_per_epoch)),
        vec(fval_per_epoch),
        color = "#4a4a4a",
    )
end
legend()
plt.gca().yaxis.set_major_formatter(sfmt)
ylabel("KL divergence + const.")
xlabel("Epochs")
legend(loc = "upper right", ncol = 1, fontsize = 9)
wsave(joinpath(save_path, "average_log.png"), fig)
close(fig)

fig = figure(figsize = (7, 4))
if epoch == args["max_epoch"]
    plot(range(0, epoch, length = length(mval)), mval, color = "#4a4a4a")
else
    plot(
        range(0, epoch, length = length(mval[1:findfirst(mval .== 0.0f0)-1])),
        mval[1:findfirst(mval .== 0.0f0)-1],
        color = "#4a4a4a",
    )
end
title("Latent distribution correction")
ylabel(
    L"$\left \| \left \| f(\mathbf{z}; \mathbf{y}) - \mathbf{x}^{\ast} \right \| \right \|_2, \quad$" *
    L"$\mathbf{z} \sim \mathrm{N}(\mathbf{z} \mid \mathbf{0}, \mathbf{I})$",
)
xlabel("Number of passes through shots")
wsave(joinpath(save_path, "err_log.png"), fig)
close(fig)


fig = figure("snr per sample size", figsize = (4, 5))
semilogx(1:test_size, snr_list_cum_amortized_mean, color = "#4a4a4a", label = "Amortized")
semilogx(
    1:test_size,
    snr_list_cum_mean,
    color = "#a1a1a1",
    label = "Latent dist. corrected",
)
title("SNR vs posterior samples size")
ylabel("Conditional mean SNR (dB)")
xlabel("Numebr of drawn posterior samples")
legend()
grid(true, which = "both", axis = "both")
wsave(joinpath(save_path, string(idx), "snr_vs_num_samples.png"), fig)
close(fig)

# Plot the true model
fig = figure("x", figsize = (7.68, 4.8))
imshow(
    X_fixed[:, :, 1, 1],
    vmin = -1.5e3,
    vmax = 1.5e3,
    aspect = 1,
    cmap = "Greys",
    resample = true,
    interpolation = "lanczos",
    filterrad = 1,
    extent = extent,
)
title("Ground-truth (unknown) image")
colorbar(fraction = 0.03, pad = 0.01, format = sfmt)
grid(false)
xlabel("Horizontal distance (km)")
ylabel("Depth (km)")
wsave(joinpath(save_path, string(idx), "true_model.png"), fig)
close(fig)

# Plot the observed data
fig = figure("y", figsize = (7.68, 4.8))
imshow(
    Y_fixed[:, :, 1, 1] / 204,
    vmin = -9.5e3,
    vmax = 9.5e3,
    aspect = 1,
    cmap = "Greys",
    resample = true,
    interpolation = "lanczos",
    filterrad = 1,
    extent = extent,
)
title("Reverse-time migrated image")
colorbar(fraction = 0.03, pad = 0.01, format = sfmt)
grid(false)
xlabel("Horizontal distance (km)")
ylabel("Depth (km)")
wsave(joinpath(save_path, string(idx), "observed_data.png"), fig)
close(fig)

fig = figure("x_cm", figsize = (7.68, 4.8))
imshow(
    X_post_amortized_mean[:, :, 1, 1],
    vmin = -1.5e3,
    vmax = 1.5e3,
    aspect = 1,
    cmap = "Greys",
    resample = true,
    interpolation = "lanczos",
    filterrad = 1,
    extent = extent,
)
title("Conditional mean estimate (amortized)")
colorbar(fraction = 0.03, pad = 0.01, format = sfmt)
grid(false)
xlabel("Horizontal distance (km)")
ylabel("Depth (km)")
wsave(joinpath(save_path, string(idx), "amortized_conditional_mean.png"), fig)
close(fig)


fig = figure("x_cm", figsize = (7.68, 4.8))
imshow(
    X_post_mean[:, :, 1, 1],
    vmin = -1.5e3,
    vmax = 1.5e3,
    aspect = 1,
    cmap = "Greys",
    resample = true,
    interpolation = "lanczos",
    filterrad = 1,
    extent = extent,
)
title("Conditional mean estimate (corrected)")
colorbar(fraction = 0.03, pad = 0.01, format = sfmt)
grid(false)
xlabel("Horizontal distance (km)")
ylabel("Depth (km)")
wsave(joinpath(save_path, string(idx), "conditional_mean.png"), fig)
close(fig)

# Plot the pointwise standard deviation
fig = figure("x_std", figsize = (7.68, 4.8))
imshow(
    X_post_amortized_std[:, :, 1, 1],
    aspect = 1,
    cmap = "OrRd",
    resample = true,
    interpolation = "kaiser",
    filterrad = 1,
    extent = extent,
    norm = matplotlib.colors.LogNorm(vmin = 40, vmax = 240.0),
)
title("Pointwise standard deviation (amortized)")
cp = colorbar(fraction = 0.03, pad = 0.01)
grid(false)
xlabel("Horizontal distance (km)")
ylabel("Depth (km)")
wsave(joinpath(save_path, string(idx), "amortized_pointwise_std.png"), fig)
close(fig)


# Plot the pointwise standard deviation
fig = figure("x_std", figsize = (7.68, 4.8))
imshow(
    X_post_std[:, :, 1, 1],
    aspect = 1,
    cmap = "OrRd",
    resample = true,
    interpolation = "kaiser",
    filterrad = 1,
    extent = extent,
    norm = matplotlib.colors.LogNorm(vmin = 40, vmax = 240.0),
)
title("Pointwise standard deviation (corrected)")
cp = colorbar(fraction = 0.03, pad = 0.01)
grid(false)
xlabel("Horizontal distance (km)")
ylabel("Depth (km)")
wsave(joinpath(save_path, string(idx), "pointwise_std.png"), fig)
close(fig)


for ns = 1:5
    fig = figure("x_cm", figsize = (7.68, 4.8))
    imshow(
        X_post_amortized[:, :, 1, ns],
        vmin = -1.5e3,
        vmax = 1.5e3,
        aspect = 1,
        cmap = "Greys",
        resample = true,
        interpolation = "lanczos",
        filterrad = 1,
        extent = extent,
    )
    title("Posterior sample (amortized)")
    colorbar(fraction = 0.03, pad = 0.01, format = sfmt)
    grid(false)
    xlabel("Horizontal distance (km)")
    ylabel("Depth (km)")
    wsave(joinpath(save_path, string(idx), "amortized_sample_" * string(ns) * ".png"), fig)
    close(fig)

    fig = figure("x_cm", figsize = (7.68, 4.8))
    imshow(
        X_post[:, :, 1, ns],
        vmin = -1.5e3,
        vmax = 1.5e3,
        aspect = 1,
        cmap = "Greys",
        resample = true,
        interpolation = "lanczos",
        filterrad = 1,
        extent = extent,
    )
    title("Posterior sample (corrected)")
    colorbar(fraction = 0.03, pad = 0.01, format = sfmt)
    grid(false)
    xlabel("Horizontal distance (km)")
    ylabel("Depth (km)")
    wsave(joinpath(save_path, string(idx), "sample_" * string(ns) * ".png"), fig)
    close(fig)
end

# Plot the pointwise standard deviation
normalized_std_amortized, analytic_mu_amortized =
    normalize_std(X_post_amortized_mean[:, :, 1, 1], X_post_amortized_std[:, :, 1, 1])
normalized_std, analytic_mu = normalize_std(X_post_mean[:, :, 1, 1], X_post_std[:, :, 1, 1])

fig = figure("x_std", figsize = (7.68, 4.8))
imshow(
    normalized_std_amortized,
    aspect = 1,
    cmap = "OrRd",
    resample = true,
    interpolation = "hermite",
    filterrad = 1,
    extent = extent,
    norm = matplotlib.colors.LogNorm(vmin = 3.0f-2, vmax = 1.2f0),
)
title("Normalized pointwise standard deviation (amortized)")
colorbar(fraction = 0.03, pad = 0.01, ticks = [6.0f-2, 1.0f-1, 1.0f0])
grid(false)
xlabel("Horizontal distance (km)")
ylabel("Depth (km)")
wsave(joinpath(save_path, string(idx), "amortized_normalized_pointwise_std.png"), fig)
close(fig)


fig = figure("x_std", figsize = (7.68, 4.8))
imshow(
    normalized_std,
    aspect = 1,
    cmap = "OrRd",
    resample = true,
    interpolation = "hermite",
    filterrad = 1,
    extent = extent,
    norm = matplotlib.colors.LogNorm(vmin = 3.0f-2, vmax = 1.2f0),
)
title("Normalized pointwise standard deviation (corrected)")
colorbar(fraction = 0.03, pad = 0.01, ticks = [6.0f-2, 1.0f-1, 1.0f0])
grid(false)
xlabel("Horizontal distance (km)")
ylabel("Depth (km)")
wsave(joinpath(save_path, string(idx), "normalized_pointwise_std.png"), fig)
close(fig)


horiz_loc = [64, 128, 192]
for loc in horiz_loc
    fig = figure("x_cm", figsize = (8, 2.5))
    plot(
        range(0.0, stop = size(X_fixed, 2) * spacing[2] / 1e3, length = size(X_fixed, 2)),
        X_post_mean[:, loc, 1, 1],
        color = "#31BFF3",
        label = "Corrected conditional mean",
        linewidth = 1.0,
    )
    plot(
        range(0.0, stop = size(X_fixed, 2) * spacing[2] / 1e3, length = size(X_fixed, 2)),
        X_fixed[:, loc, 1, 1],
        color = "k",
        "--",
        alpha = 0.5,
        label = "Ground-trurh image",
        linewidth = 1.0,
    )
    fill_between(
        range(0.0, stop = size(X_fixed, 2) * spacing[2] / 1e3, length = size(X_fixed, 2)),
        X_post_mean[:, loc, 1, 1] - 2.576 * X_post_std[:, loc, 1, 1],
        X_post_mean[:, loc, 1, 1] + 2.576 * X_post_std[:, loc, 1, 1],
        color = "#FFAF68",
        alpha = 0.8,
        label = "%99 confidence interval",
        facecolor = "#FFAF68",
        edgecolor = "#FFAF68",
        linewidth = 0.5,
        linestyle = "solid",
    )
    ticklabel_format(axis = "y", style = "sci", useMathText = true)
    grid(true)
    leg = legend(loc = "lower left", ncol = 3, fontsize = 8)
    title("Vertical profile at " * string(loc * spacing[1] / 1e3) * " km")
    ylabel("Perturbation")
    ylim([-1400, 1300])
    xlim([-0.05, size(X_fixed, 2) * spacing[2] / 1e3 + 0.05])
    xlabel("Depth (km)")
    wsave(
        joinpath(save_path, string(idx), "vertical_profile_at_" * string(loc) * ".png"),
        fig,
    )
    close(fig)

end

# points = pick_points(dm)
points = [(60, 70), [70, 200], [200, 150]]
labels = String[]
for p in points
    label =
        "(" *
        string(p[1] * spacing[1] / 1e3) *
        " km, " *
        string(p[2] * spacing[2] / 1e3) *
        " km)"
    push!(labels, label)
end

for p_idx = 1:1:length(points)

    fig = figure("histogram", figsize = (7, 3))
    ax = distplot(
        X_prior[points[p_idx][1], points[p_idx][2], 1, :],
        bins = 40,
        kde = true,
        hist_kws = Dict(
            "density" => true,
            "label" => "Learned prior",
            "alpha" => 0.5,
            "histtype" => "bar",
        ),
        color = "#F4889A",
    )

    distplot(
        X_post_amortized[points[p_idx][1], points[p_idx][2], 1, :],
        bins = 30,
        kde = true,
        hist_kws = Dict(
            "density" => true,
            "label" => "Amortized posterior",
            "alpha" => 0.8,
            "histtype" => "bar",
        ),
        color = "#79D45E",
    )

    distplot(
        X_post[points[p_idx][1], points[p_idx][2], 1, :],
        bins = 15,
        kde = true,
        hist_kws = Dict(
            "density" => true,
            "label" => "Correcred posterior",
            "alpha" => 0.5,
            "histtype" => "bar",
        ),
        color = "#31BFF3",
    )

    axvline(
        X_fixed[points[p_idx][1], points[p_idx][2], 1, 1],
        linestyle = "solid",
        linewidth = 1.8,
        label = "Ground-truth value",
        color = "#000000",
    )

    axvline(
        X_post_mean[points[p_idx][1], points[p_idx][2], 1, 1],
        linestyle = "--",
        linewidth = 1.8,
        label = "Corrected conditional mean",
        color = "#000000",
    )

    for label in ax.get_xticklabels()
        label.set_fontproperties(font_prop)
    end

    for label in ax.get_yticklabels()
        label.set_fontproperties(font_prop)
    end

    xlabel("Perturbation", fontproperties = font_prop)
    ylabel("Probability density function", fontproperties = font_prop)
    xlim([-1000, 1000])
    grid(true)
    # ylim([0, 125])
    title("Pointwise histograms at " * labels[p_idx])
    ax.legend(loc = "upper right", fontsize = 7.5)
    wsave(
        joinpath(
            save_path,
            string(idx),
            "histogram-at-" *
            string(points[p_idx][1]) *
            "-" *
            string(points[p_idx][2]) *
            ".png",
        ),
        fig,
    )
    close(fig)

end



(J, Mr), noise_dist = create_operator(; nrec = nrec, nsrc = nsrc, sigma = sigma)
src_idx = Int(ceil(nsrc / 2))
d_obs = J[src_idx] * Mr * vec(convert(Array{Float32}, X_fixed[:, :, 1, 1]'))
d_pred_amortized =
    J[src_idx] * Mr * vec(convert(Array{Float32}, X_post_amortized_mean[:, :, 1, 1]'))
d_pred = J[src_idx] * Mr * vec(convert(Array{Float32}, X_post_mean[:, :, 1, 1]'))

d_true = reshape(d_obs, (:, nrec))
d_obs = reshape(d_obs + rand(noise_dist), (:, nrec))
d_pred_amortized = reshape(d_pred_amortized, (:, nrec))
d_pred = reshape(d_pred, (:, nrec))

zero_off_loc =
    range(0.0f0, stop = J.model.d[1] * (J.model.n[1] - 1.0f0), length = nsrc)[src_idx]
extent_d = [
    -zero_off_loc / 1.0e3,
    (J.model.d[1] * (J.model.n[1] - 1) - zero_off_loc) / 1.0e3,
    2000.0f0 / 1.0e3,
    0.0,
]

rc("font", family = "serif", size = 9)
font_prop =
    matplotlib.font_manager.FontProperties(family = "serif", style = "normal", size = 8)

# Plotting single src. noise free observed data
fig = figure("d_true", figsize = (4, 7))
imshow(
    d_true,
    vmin = -7.0f3,
    vmax = 7.0f3,
    extent = extent_d,
    cmap = "Greys",
    alpha = 1.0,
    aspect = 5.0,
    resample = true,
    interpolation = "hanning",
    filterrad = 1,
)
title("Noise free data")
grid(false)
xlabel("Offset (km)")
ylabel("Time (s)")
wsave(joinpath(save_path, string(idx), "d_noise_free.png"), fig)
close(fig)

# Plotting single src. noise free observed data
fig = figure("d_obs", figsize = (4, 7))
imshow(
    d_obs,
    vmin = -7.0f3,
    vmax = 7.0f3,
    extent = extent_d,
    cmap = "Greys",
    alpha = 1.0,
    aspect = 5.0,
    resample = true,
    interpolation = "hanning",
    filterrad = 1,
)
title("Observed data")
grid(false)
xlabel("Offset (km)")
ylabel("Time (s)")
wsave(joinpath(save_path, string(idx), "d_obs.png"), fig)
close(fig)

# Plotting single src. noise free observed data
fig = figure("d_pred_amortized", figsize = (4, 7))
imshow(
    d_pred_amortized,
    vmin = -7.0f3,
    vmax = 7.0f3,
    extent = extent_d,
    cmap = "Greys",
    alpha = 1.0,
    aspect = 5.0,
    resample = true,
    interpolation = "hanning",
    filterrad = 1,
)
title("Predicted data (amortized)")
grid(false)
xlabel("Offset (km)")
ylabel("Time (s)")
wsave(joinpath(save_path, string(idx), "d_pred_amortized.png"), fig)
close(fig)

# Plotting single src. noise free observed data
fig = figure("d_pred", figsize = (4, 7))
imshow(
    d_pred,
    vmin = -7.0f3,
    vmax = 7.0f3,
    extent = extent_d,
    cmap = "Greys",
    alpha = 1.0,
    aspect = 5.0,
    resample = true,
    interpolation = "hanning",
    filterrad = 1,
)
title("Predicted data (corrected)")
grid(false)
xlabel("Offset (km)")
ylabel("Time (s)")
wsave(joinpath(save_path, string(idx), "d_pred.png"), fig)
close(fig)

# Plotting single src. noise free observed data
fig = figure("error", figsize = (4, 7))
imshow(
    d_pred_amortized - d_pred,
    vmin = -7.0f3,
    vmax = 7.0f3,
    extent = extent_d,
    cmap = "Greys",
    alpha = 1.0,
    aspect = 5.0,
    resample = true,
    interpolation = "hanning",
    filterrad = 1,
)
title("Data prediction error (amortized)")
grid(false)
xlabel("Offset (km)")
ylabel("Time (s)")
wsave(joinpath(save_path, string(idx), "d_pred_amortized_error.png"), fig)
close(fig)

# Plotting single src. noise free observed data
fig = figure("error", figsize = (4, 7))
imshow(
    d_true - d_pred,
    vmin = -7.0f3,
    vmax = 7.0f3,
    extent = extent_d,
    cmap = "Greys",
    alpha = 1.0,
    aspect = 5.0,
    resample = true,
    interpolation = "hanning",
    filterrad = 1,
)
title("Data prediction error (corrected)")
grid(false)
xlabel("Offset (km)")
ylabel("Time (s)")
wsave(joinpath(save_path, string(idx), "d_pred_error.png"), fig)
close(fig)

fig = figure("profile", figsize = (3, 7))
plot(
    d_true[:, Int(floor(nrec / 2))],
    range(0.0, extent_d[3], length = size(d_true, 1)),
    color = "#000000",
    lw = 0.5,
    alpha = 1.0,
    label = "Corrected prediction",
)
plot(
    d_pred_amortized[:, Int(floor(nrec / 2))],
    range(0.0, extent_d[3], length = size(d_true, 1)),
    color = "#a1a1a1",
    lw = 0.8,
    alpha = 1.0,
    label = "Amortized prediction",
    "--",
)
plot(
    d_pred[:, Int(floor(nrec / 2))],
    range(0.0, extent_d[3], length = size(d_true, 1)),
    color = "#4a4a4a",
    lw = 1,
    alpha = 1.0,
    label = "Corrected prediction",
    ":",
)
gca().invert_yaxis()
grid(true)
leg = legend(loc = "upper left", ncol = 1, fontsize = 8)
leg.get_lines()[1].set_linewidth(2.5)
xlim([-4.0f3, 4.0f3])
title("Zero-offset trace")
xlabel("Perturbation")
ylim([2.0, -0.1])
ylabel("Time (s)")
wsave(joinpath(save_path, string(idx), "pred_error_trace.png"), fig)
close(fig)

fig = figure("profile", figsize = (3, 7))
plot(
    d_true[:, nrec-5],
    range(0.0, extent_d[3], length = size(d_true, 1)),
    color = "#000000",
    lw = 0.5,
    alpha = 1.0,
    label = "Corrected prediction",
)
plot(
    d_pred_amortized[:, nrec-5],
    range(0.0, extent_d[3], length = size(d_true, 1)),
    color = "#a1a1a1",
    lw = 0.8,
    alpha = 1.0,
    label = "Amortized prediction",
    "--",
)
plot(
    d_pred[:, nrec-5],
    range(0.0, extent_d[3], length = size(d_true, 1)),
    color = "#4a4a4a",
    lw = 1,
    alpha = 1.0,
    label = "Corrected prediction",
    ":",
)
gca().invert_yaxis()
grid(true)
leg = legend(loc = "upper left", ncol = 1, fontsize = 8)
leg.get_lines()[1].set_linewidth(2.5)
xlim([-6.0f3, 6.0f3])
title(L"$2.4\,$km$-$offset trace")
xlabel("Perturbation")
ylim([2.0, 1 - 0.1])
ylabel("Time (s)")
wsave(joinpath(save_path, string(idx), "pred_error_trace_far_offset.png"), fig)
close(fig)

println("SNR of amortized conditional mean: ", X_post_amortized_mean_snr)
println("SNR of conditional mean: ", X_post_mean_snr)
println("SNR of RTM image: ", Y_fixed_snr)

open(joinpath(save_path, string(idx), "snr-values-amortized.txt"), "w") do io
    write(io, "SNR of conditional mean: ", string(X_post_amortized_mean_snr), "\n")
    write(io, "SNR of RTM image: ", string(Y_fixed_snr), "\n")

    for j = 1:test_size
        write(io, "SNR of sample ", string(j), ": ", string(snr_list_amortized[j]), "\n")
    end
end

open(joinpath(save_path, string(idx), "snr-values.txt"), "w") do io
    write(io, "SNR of conditional mean: ", string(X_post_mean_snr), "\n")
    write(io, "SNR of RTM image: ", string(Y_fixed_snr), "\n")

    for j = 1:test_size
        write(io, "SNR of sample ", string(j), ": ", string(snr_list[j]), "\n")
    end
end
