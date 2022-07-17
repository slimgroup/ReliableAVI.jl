using DrWatson
@quickactivate :ReliableAVI

using JLD
using HDF5
using PyPlot
using ArgParse
using ProgressMeter

function create_OOD_shots_and_RTM(args::Dict)

    sim_name = "create_OOD_shots_and_RTM"
    data_args = Dict(
        "nsrc" => args["nsrc"],
        "nrec" => args["nrec"],
        "sigma" => args["sigma"],
        "parihaka_label" => args["parihaka_label"],
        "idx" => args["idx"],
        "sim_name" => sim_name,
    )

    dm, lin_data, rtm = load_shot_data(data_args)

    if (dm == lin_data == rtm == nothing)

        sigma = args["sigma"]
        idx = args["idx"]
        parihaka_label = args["parihaka_label"]
        nsrc = args["nsrc"]
        nrec = args["nrec"]

        # Define raw data directory
        mkpath(datadir("training-data"))

        if parihaka_label == 1
            data_path = datadir("training-data", "training-pairs.h5")

            # Download the dataset into the data directory if it does not exist
            if isfile(data_path) == false
                run(`wget https://www.dropbox.com/s/53u8ckb9aje8xv4/'
                    'training-pairs.h5 -q -O $data_path`)
            end

            # Load seismic images and create training and testing data
            file = h5open(data_path, "r")
            X_fixed = file["dm"][:, :, :, idx:idx]
            close(file)
        else
            data_path = datadir("training-data", "seismic_samples_256_by_256_num_10k.jld")
            label_path =
                datadir("training-data", "seismic_samples_256_by_256_num_10k_labels.jld")

            # Download the dataset into the data directory if it does not exist
            if isfile(data_path) == false
                run(`wget https://www.dropbox.com/s/vapl62yhh8fxgwy/'
                    'seismic_samples_256_by_256_num_10k.jld -q -O $data_path`)
            end
            if isfile(label_path) == false
                run(`wget https://www.dropbox.com/s/blxkh6tdszudlcq/'
                    'seismic_samples_256_by_256_num_10k_labels.jld -q -O $label_path`)
            end

            # Load OOD seismic images
            X_fixed = JLD.jldopen(data_path, "r") do file
                X_fixed = read(file, "X")

                labels = JLD.jldopen(label_path, "r")["labels"][:]
                idx_label = findall(x -> x == parihaka_label, labels)[idx]
                return X_fixed[:, :, :, idx_label:idx_label]
            end
        end
        dm = X_fixed[:, :, 1, 1]'
        dm[:, 1:10] .= 0.0
        dm = convert(Array{Float32,1}, vec(dm))

        # Create forward modeling operator
        (J, Mr), noise_dist =
            create_operator(; nrec = nrec, nsrc = nsrc, sigma = sigma, sim_src = false)
        lin_data = zeros(Float32, nsrc, J.source.geometry.nt[1] * nrec)
        rtm = zeros(Float32, size(dm))

        p = Progress(nsrc)
        for j = 1:nsrc
            Base.flush(Base.stdout)

            lin_data[j, :] = J[j] * dm + rand(noise_dist)
            grad = adjoint(J[j]) * lin_data[j, :]
            rtm = (rtm * (j - 1) + grad) / j

            ProgressMeter.next!(p; showvalues = [(:source, j)])
        end

        dm = reshape(
            reshape(dm, size(X_fixed)[1:2])',
            size(X_fixed)[1],
            size(X_fixed)[2],
            1,
            1,
        )
        # 204 is the number of receivers used during training. Renormalizing the amplitudes
        # according to that.
        rtm =
            convert(Float32, 204 / nrec) * reshape(
                reshape(rtm, size(X_fixed)[1:2])',
                size(X_fixed)[1],
                size(X_fixed)[2],
                1,
                1,
            )

        save_dict = @strdict sigma idx nsrc nrec dm lin_data rtm sim_name parihaka_label
        @tagsave(
            datadir(sim_name, savename(save_dict, "jld2"; digits = 6)),
            save_dict;
            safe = true
        )
    end
    return dm, lin_data, rtm
end
