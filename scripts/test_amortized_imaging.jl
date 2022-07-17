# Authors: Ali Siahkoohi, alisk@gatech.edu
# Date: March 2022


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

font_prop, sfmt = sef_plot_configs()
args = read_config("test_amortized_imaging.json")
args = parse_input_args(args)


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
lr_step = args["lr_step"]
batchsize = args["batchsize"]
n_hidden = args["n_hidden"]
depth = args["depth"]
sim_name = args["sim_name"]
if args["epoch"] == -1
    args["epoch"] = args["max_epoch"]
end
epoch = args["epoch"]

# Define raw data directory
mkpath(datadir("training-data"))
data_path = datadir("training-data", "training-pairs.h5")

# Download the dataset into the data directory if it does not exist
if isfile(data_path) == false
    run(`wget https://www.dropbox.com/s/53u8ckb9aje8xv4/'
        'training-pairs.h5 -q -O $data_path`)
end

# Load seismic images and create training and testing data
file = h5open(data_path, "r")
X_train = file["dm"][:, :, :, :]
Y_train = file["rtm"][:, :, :, :]
Y_train[1:10, :, :, :] .= 0.0f0

nx, ny, nc, nsamples = size(X_train)

AN_x = ActNorm(nsamples)
AN_y = ActNorm(nsamples)
X_train = AN_x.forward(X_train)
Y_train = AN_y.forward(Y_train)

# Dimensions after wavelet squeeze to increase no. of channels
nx = Int(nx / 2)
ny = Int(ny / 2)
n_in = Int(nc * 4)

# Create network
CH = NetworkConditionalHINT(n_in, n_hidden, depth)

# Loading the experiment—only network weights and training loss
loaded_keys = load_experiment(args, ["Params", "fval", "fval_eval", "train_idx"])
Params = loaded_keys["Params"]
fval = loaded_keys["fval"]
fval_eval = loaded_keys["fval_eval"]
train_idx = loaded_keys["train_idx"]
put_params!(CH, convert(Array{Any,1}, Params))

# test data pairs
idx = 50 #shuffle(setdiff(1:nsamples, train_idx))[1] #4009
X_fixed = wavelet_squeeze(X_train[:, :, :, idx:idx])
Y_fixed = wavelet_squeeze(Y_train[:, :, :, idx:idx])


# Now select single fixed sample from all Ys
Zy_fixed = CH.forward_Y(Y_fixed)

# Draw new Zx, while keeping Zy fixed
n_samples = 1000
X_post = zeros(Float32, nx, ny, n_in, n_samples)
CH = CH |> gpu

test_batchsize = 4
test_loader = Flux.DataLoader(
    (randn(Float32, nx, ny, n_in, n_samples), repeat(Zy_fixed, 1, 1, 1, n_samples)),
    batchsize = test_batchsize,
    shuffle = false,
)

p = Progress(length(test_loader))

for (itr, (X, Y)) in enumerate(test_loader)
    Base.flush(Base.stdout)
    counter = (itr - 1) * test_batchsize + 1

    X = X |> gpu
    Y = Y |> gpu

    X_post[:, :, :, counter:(counter+size(X)[4]-1)] = (CH.inverse(X, Y)[1] |> cpu)
    ProgressMeter.next!(p)
end

X_post = wavelet_unsqueeze(X_post)
X_post = AN_x.inverse(X_post)
X_post[1:10, :, :, :] .= 0.0f0

# Some stats
X_post_std = std(X_post; dims = 4)

X_post_cum_mean =
    cumsum(X_post, dims = 4) ./
    reshape(collect(1:size(X_post)[4]), (1, 1, 1, size(X_post)[4]))
X_post_mean = X_post_cum_mean[:, :, :, end:end]

X_fixed = wavelet_unsqueeze(X_fixed)
Y_fixed = wavelet_unsqueeze(Y_fixed)
Zy_fixed = wavelet_unsqueeze(Zy_fixed)

X_fixed = AN_x.inverse(X_fixed)
Y_fixed = AN_y.inverse(Y_fixed)
Y_fixed[1:10, :, :, :] .= 0.0f0

save_dict = @strdict max_epoch epoch lr lr_step batchsize n_hidden depth sim_name
save_path = plotsdir(sim_name, savename(save_dict; digits = 6))

spacing = [20.0, 12.5]
extent = [0.0, size(X_fixed, 1) * spacing[1], size(X_fixed, 2) * spacing[2], 0.0] / 1e3

signal_to_noise(xhat, x) = -20.0 * log(norm(x - xhat) / norm(x)) / log(10.0)

snr_list = []
snr_list_cum_mean = []
for j = 1:n_samples
    push!(snr_list, signal_to_noise(X_post[:, :, :, j], X_fixed[:, :, :, 1]))
    push!(
        snr_list_cum_mean,
        signal_to_noise(X_post_cum_mean[:, :, :, j], X_fixed[:, :, :, 1]),
    )
end

X_post_mean_snr = signal_to_noise(X_post_mean[:, :, :, 1], X_fixed[:, :, :, 1])
Y_fixed_snr = signal_to_noise(Y_fixed[:, :, :, 1] / 204, X_fixed[:, :, :, 1])

# Training loss
fig = figure("training logs", figsize = (7, 4))
if epoch == args["max_epoch"]
    plot(
        range(0, epoch, length = length(fval_eval)),
        fval_eval,
        color = "#4a4a4a",
        label = "validation loss",
    )
else
    plot(
        range(0, epoch, length = length(fval_eval[1:findfirst(fval_eval .== 0.0f0)-1])),
        fval_eval[1:findfirst(fval_eval .== 0.0f0)-1],
        color = "#4a4a4a",
        label = "validation loss",
    )
end
ticklabel_format(axis = "y", style = "sci", useMathText = true)
title("Negative log-likelihood")
ylabel("Validation objective")
xlabel("Epochs")
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
        label = "Training",
    )
    plot(
        range(0, epoch, length = length(fval_eval)),
        fval_eval,
        "--",
        color = "#a1a1a1",
        label = "Validation",
    )
else
    fval_per_epoch = fval[1:findfirst(fval .== 0.0f0)-1]
    fval_per_epoch =
        mean(reshape(fval_per_epoch, (Int(length(fval) / length(fval_eval))), :), dims = 1)
    plot(
        range(0, epoch, length = length(fval_per_epoch)),
        vec(fval_per_epoch),
        color = "#4a4a4a",
        label = "Training",
    )
    plot(
        range(0, epoch, length = length(fval_eval[1:findfirst(fval_eval .== 0.0f0)-1])),
        fval_eval[1:findfirst(fval_eval .== 0.0f0)-1],
        "--",
        color = "#a1a1a1",
        label = "Validation",
    )
end
legend()
plt.gca().yaxis.set_major_formatter(sfmt)
# plt.gca().xaxis.set_major_formatter(sfmt)
# title("Amortized variational inference objective")
ylabel("Average KL divergence + const.")
xlabel("Epochs")
legend(loc = "upper right", ncol = 1, fontsize = 9)
wsave(joinpath(save_path, "average_log.png"), fig)
close(fig)


fig = figure("snr per sample size", figsize = (4, 7))
semilogx(1:n_samples, snr_list_cum_mean, color = "#000000")
# title("Conditional mean SNR")
ylabel("Conditional mean signal-to-noise ratio (dB)")
xlabel("Numebr of posterior samples")
grid(true, which = "both", axis = "both")
wsave(joinpath(save_path, string(idx), "snr_vs_num_samples.png"), fig)
close(fig)


v0 = ones(Float32, size(X_fixed)) * 2.5f0
v0 .*= reshape(
    range(1.0f0, 4.5f0 / 2.5f0, length = size(X_fixed, 2)),
    (size(X_fixed, 2), 1, 1, 1),
)
s = 1.0f0 ./ v0
s[1:9, :] .= 1.0f0 / 1.5f0
m0 = s .^ 2.0f0

fig = figure("m0", figsize = (7.68, 4.8))
imshow(
    m0[:, :, 1, 1],
    vmin = 0.0,
    vmax = 0.2,
    aspect = 1,
    cmap = "Greys",
    resample = true,
    interpolation = "kaiser",
    filterrad = 1,
    extent = extent,
)
grid(false)
# title("Smooth background model")
cb = colorbar(fraction = 0.03, pad = 0.01, format = sfmt, ticks = 0.0f0:0.5f-1:2.4f-1)
cb.set_label(label = L"$\frac{\mathrm{s}^2}{\mathrm{km}^2}$", fontsize = 12)
grid(false)
xlabel("Horizontal distance (km)")
ylabel("Depth (km)")
wsave(joinpath(save_path, string(idx), "background.png"), fig)
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

# Plot the conditional mean estimate
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
title("Conditional mean estimate (amortized)")
colorbar(fraction = 0.03, pad = 0.01, format = sfmt)
grid(false)
xlabel("Horizontal distance (km)")
ylabel("Depth (km)")
wsave(joinpath(save_path, string(idx), "conditional_mean.png"), fig)
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
title("Pointwise standard deviation (amortized)")
cp = colorbar(fraction = 0.03, pad = 0.01)
grid(false)
xlabel("Horizontal distance (km)")
ylabel("Depth (km)")
wsave(joinpath(save_path, string(idx), "pointwise_std.png"), fig)
close(fig)

for ns = 1:10
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
    title("Posterior sample (amortized)")
    colorbar(fraction = 0.03, pad = 0.01, format = sfmt)
    grid(false)
    xlabel("Horizontal distance (km)")
    ylabel("Depth (km)")
    wsave(joinpath(save_path, string(idx), "sample_" * string(ns) * ".png"), fig)
    close(fig)
end

# Plot the pointwise standard deviation
normalized_std, analytic_mu = normalize_std(X_post_mean[:, :, 1, 1], X_post_std[:, :, 1, 1])

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
title("Normalized pointwise standard deviation (amortized)")
colorbar(fraction = 0.03, pad = 0.01, ticks = [6.0f-2, 1.0f-1, 1.0f0])
grid(false)
xlabel("Horizontal distance (km)")
ylabel("Depth (km)")
wsave(joinpath(save_path, string(idx), "normalized_pointwise_std.png"), fig)
close(fig)


nrec = 204
nsrc = 102
sigma = 2000.0f0

(J, Mr), noise_dist = create_operator(; nrec = nrec, nsrc = nsrc, sigma = sigma)
src_idx = Int(ceil(nsrc / 2))
d_obs = J[src_idx] * Mr * vec(convert(Array{Float32}, X_fixed[:, :, 1, 1]'))
d_pred = J[src_idx] * Mr * vec(convert(Array{Float32}, X_post_mean[:, :, 1, 1]'))

d_true = reshape(d_obs, (:, nrec))
d_obs = reshape(d_obs + rand(noise_dist), (:, nrec))
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
title("Predicted data (amortized)")
grid(false)
xlabel("Offset (km)")
ylabel("Time (s)")
wsave(joinpath(save_path, string(idx), "d_pred.png"), fig)
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
title("Data prediction error (amortized)")
grid(false)
xlabel("Offset (km)")
ylabel("Time (s)")
wsave(joinpath(save_path, string(idx), "d_pred_error.png"), fig)
close(fig)

println("SNR of conditional mean: ", X_post_mean_snr)
println("SNR of RTM image: ", Y_fixed_snr)

open(joinpath(save_path, string(idx), "snr-values.txt"), "w") do io
    write(io, "SNR of conditional mean: ", string(X_post_mean_snr), "\n")
    write(io, "SNR of RTM image: ", string(Y_fixed_snr), "\n")

    for j = 1:n_samples
        write(io, "SNR of sample ", string(j), ": ", string(snr_list[j]), "\n")
    end
end
