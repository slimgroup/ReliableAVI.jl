# Authors: Ali Siahkoohi, alisk@gatech.edu
# Date: March 2022

using DrWatson
@quickactivate :ReliableAVI

using InvertibleNetworks
using HDF5
using Random
using Statistics
using ProgressMeter
using Flux

# Random seed
Random.seed!(19)

args = read_config("train_amortized_imaging.json")
args = parse_input_args(args)

max_epoch = args["max_epoch"]
lr = args["lr"]
lr_step = args["lr_step"]
batchsize = args["batchsize"]
n_hidden = args["n_hidden"]
depth = args["depth"]
sim_name = args["sim_name"]
resume = args["resume"]

# Loading the existing weights, if any.
loaded_keys = resume_from_checkpoint(args, ["Params", "fval", "fval_eval", "opt", "epoch"])
Params = loaded_keys["Params"]
fval = loaded_keys["fval"]
fval_eval = loaded_keys["fval_eval"]
opt = loaded_keys["opt"]
init_epoch = loaded_keys["epoch"]

# Define raw data directory
mkpath(datadir("training-data"))
data_path = datadir("training-data", "training-pairs.h5")

# Download the dataset into the data directory if it does not exist
if isfile(data_path) == false
    run(`wget https://www.dropbox.com/s/53u8ckb9aje8xv4/'
        'training-pairs.h5 --no-check-certificate -q -O $data_path`)
end

# Load seismic images and create training and testing data
file = h5open(data_path, "r")
X_train = file["dm"][:, :, :, :]
Y_train = file["rtm"][:, :, :, :]
close(file)

Y_train[1:10, :, :, :] .= 0.0f0

nx, ny, nc, nsamples = size(X_train)

AN_x = ActNorm(nsamples)
AN_y = ActNorm(nsamples)
X_train = AN_x.forward(X_train)
Y_train = AN_y.forward(Y_train)

AN_params_x = get_params(AN_x)
AN_params_y = get_params(AN_y)

# Split in training/validation
ntrain = Int(floor((nsamples * 0.9)))
train_idx = randperm(nsamples)[1:ntrain]
val_idx = shuffle(setdiff(1:nsamples, train_idx))[1:64]

# Dimensions after wavelet squeeze to increase no. of channels
nx = Int(nx / 2)
ny = Int(ny / 2)
n_in = Int(nc * 4)


# Create network
CH = NetworkConditionalHINT(n_in, n_hidden, depth)
Params != nothing && put_params!(CH, convert(Array{Any,1}, Params))
CH = CH |> gpu

X_val = wavelet_squeeze(X_train[:, :, :, val_idx]) |> gpu
Y_val = wavelet_squeeze(Y_train[:, :, :, val_idx]) |> gpu

# Training
# Batch extractor
train_loader = Flux.DataLoader(train_idx, batchsize = batchsize, shuffle = true)
num_batches = length(train_loader)

# Optimizer
opt == nothing && (
    opt = Flux.Optimiser(
        Flux.ExpDecay(lr, 9.0f-1, num_batches * lr_step, 1.0f-6),
        Flux.ADAM(lr),
    )
)

# Training log keeper
fval == nothing && (fval = zeros(Float32, num_batches * max_epoch))
fval_eval == nothing && (fval_eval = zeros(Float32, max_epoch))

p = Progress(num_batches * (max_epoch - init_epoch + 1))

for epoch = init_epoch:max_epoch

    fval_eval[epoch] = loss_supervised(CH, X_val, Y_val; grad = false)

    for (itr, idx) in enumerate(train_loader)
        Base.flush(Base.stdout)

        # Augmentation
        if rand() > 5.0f-1
            X = X_train[:, end:-1:1, :, idx]
            Y = Y_train[:, end:-1:1, :, idx]
        else
            X = X_train[:, :, :, idx]
            Y = Y_train[:, :, :, idx]
        end

        # Apply wavelet squeeze.
        X = wavelet_squeeze(X)
        Y = wavelet_squeeze(Y)

        X = X |> gpu
        Y = Y |> gpu

        fval[(epoch-1)*num_batches+itr] = loss_supervised(CH, X, Y)[1]

        ProgressMeter.next!(
            p;
            showvalues = [
                (:Epoch, epoch),
                (:Itreration, itr),
                (:NLL, fval[(epoch-1)*num_batches+itr]),
                (:NLL_eval, fval_eval[epoch]),
            ],
        )

        # Update params
        for p in get_params(CH)
            Flux.update!(opt, p.data, p.grad)
        end
        clear_grad!(CH)
    end

    # Saving parameters and logs
    Params = get_params(CH) |> cpu
    save_dict =
        @strdict epoch max_epoch lr lr_step batchsize n_hidden depth sim_name Params fval fval_eval train_idx AN_params_x AN_params_y opt
    @tagsave(
        datadir(sim_name, savename(save_dict, "jld2"; digits = 6)),
        save_dict;
        safe = true
    )
end
