using DrWatson
@quickactivate :ReliableAVI

using InvertibleNetworks
using ProgressMeter
using LinearAlgebra
using Random
using Statistics
using Flux
include(scriptsdir(joinpath("seismic-imaging", "create_OOD_shots_and_RTM.jl")))

# Random seed
Random.seed!(19)

# Configure device
device = cpu

args = read_config("finetune_amortized_imaging.json")
args = parse_input_args(args)

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
resume = args["resume"]

# Loading the existing weights, if any.
loaded_keys = resume_from_checkpoint(
    args,
    ["Params", "J", "Mr", "src_enc", "Z0_val", "opt", "epoch", "fval", "mval", "fval_eval"],
)
Params, J, Mr, src_enc, Z0_val, opt, init_epoch, fval, mval, fval_eval =
    loaded_keys["Params"],
    loaded_keys["J"],
    loaded_keys["Mr"],
    loaded_keys["src_enc"],
    loaded_keys["Z0_val"],
    loaded_keys["opt"],
    loaded_keys["epoch"],
    loaded_keys["fval"],
    loaded_keys["mval"],
    loaded_keys["fval_eval"]

# Create simultaneous source operators and return the source encoding.
if (J == Mr == nothing)
    (J, Mr), _, src_enc =
        create_operator(; nrec = nrec, nsrc = nsrc, sigma = sigma, sim_src = true)
end

# Create shot data and RTM image given a index for out-of-distribution seismic image.
X_OOD, lin_data, Y_OOD = create_OOD_shots_and_RTM(args)
Y_OOD[1:10, :, :, :] .= 0.0f0

# Mix the data according to the source encoding to create simultaneous data.
mix_data!(lin_data, src_enc)

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

X_OOD = AN_x.forward(X_OOD)
Y_OOD = AN_y.forward(Y_OOD)
X_OOD = wavelet_squeeze(X_OOD)
Y_OOD = wavelet_squeeze(Y_OOD)

nx, ny, n_in = size(X_OOD)[1:3]

# Define network mapping its latent space to the pretrained model's latent space.
H = ActNorm(nx * ny * n_in, logdet = true)
H.b.data = zeros(Float32, nx * ny * n_in)
H.s.data = ones(Float32, nx * ny * n_in)
Params != nothing && put_params!(H, convert(Array{Any,1}, Params))
H = H |> device

# Define the amortized network
CH = NetworkConditionalHINT(n_in, n_hidden_amortized, depth_amortized, logdet = false)
put_params!(CH, convert(Array{Any,1}, Params_amortized))
CH = CH |> device

# Corresponding data latent variable
Y_OOD = Y_OOD |> device
X_OOD = X_OOD |> device
Zy_obs = CH.forward_Y(Y_OOD)
CHrev = reverse(CH)
AN_x_rev = reverse(deepcopy(AN_x))

# Validation latent variables.
nval = 4
Z0_val == nothing && (Z0_val = randn(Float32, nx, ny, n_in, nval))
Z0_val = Z0_val |> device

# Training Batch extractor
train_loader = Flux.DataLoader(
    range(1, nsrc, step = 1),
    batchsize = 1,
    shuffle = true,
    partial = false,
)
num_batches = length(train_loader)

# Optimizer
opt == nothing && (
    opt = Flux.Optimiser(
        Flux.ExpDecay(lr, 0.9f0, num_batches * lr_step, 1.0f-6),
        Flux.ADAM(lr),
    )
)

# Training log keeper.
fval == nothing && (fval = zeros(Float32, num_batches * max_epoch))
mval == nothing && (mval = zeros(Float32, num_batches * max_epoch))
fval_eval == nothing && (fval_eval = zeros(Float32, max_epoch))

p = Progress(num_batches * (max_epoch - init_epoch + 1))
src_list = collect(1:nsrc)

for epoch = init_epoch:max_epoch
    shuffle!(src_list)
    for j = 1:nval
        fval_eval[epoch] += loss_unsupervised_finetuning(
            H,
            CHrev,
            lin_data[[11]..., :],
            Z0_val[:, :, :, j:j],
            Zy_obs,
            J[[11]...],
            Mr,
            sigma,
            sigma_prior,
            AN_x_rev,
            nsrc,
            device;
            grad = false,
        )[1]
    end
    fval_eval[epoch] /= nval

    for (itr, idx) in enumerate(train_loader)
        Base.flush(Base.stdout)
        srx_idx = getindex(src_list, idx)

        fval[(epoch-1)*num_batches+itr], X_hat = loss_unsupervised_finetuning(
            H,
            CHrev,
            lin_data[srx_idx..., :],
            randn(Float32, nx, ny, n_in, 1) |> device,
            Zy_obs,
            J[srx_idx...],
            Mr,
            sigma,
            sigma_prior,
            AN_x_rev,
            nsrc,
            device,
        )

        mval[(epoch-1)*num_batches+itr] = norm(X_hat - X_OOD)^2 / length(X_hat)

        ProgressMeter.next!(
            p;
            showvalues = [
                (:Epoch, epoch),
                (:Itreration, itr),
                (:NLL, fval[(epoch-1)*num_batches+itr]),
                (:NLL_eval, fval_eval[epoch]),
                (:Error, mval[(epoch-1)*num_batches+itr]),
                (:mean_s_data, mean(H.s.data)),
                (:mean_s_grad, mean(H.s.grad)),
                (:mean_b_data, mean(H.b.data)),
                (:mean_b_grad, mean(H.b.grad)),
            ],
        )

        # Update params
        for p in get_params(H)
            Flux.update!(opt, p.data, p.grad)
        end
        clear_grad!(H)
    end
    # Saving parameters and logs
    local Params = get_params(H)
    save_dict =
        @strdict epoch max_epoch lr lr_step sigma sigma_prior sim_name Params fval mval fval_eval opt J Mr Z0_val amortized_model_config nsrc nrec src_enc idx parihaka_label
    @tagsave(
        datadir(sim_name, savename(save_dict, "jld2"; digits = 6)),
        save_dict;
        safe = true
    )
end
