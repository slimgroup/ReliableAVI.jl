# Author: Ali Siahkoohi, alisk@gatech.edu
# Date: March 2022
# Copyright: Georgia Institute of Technology, 2022

export _wsave, sef_plot_configs, wiggle_plot

_wsave(s, fig::Figure; dpi::Int = 200) = fig.savefig(s, bbox_inches = "tight", dpi = dpi)


function sef_plot_configs(; fontsize = 10)
    set_style("whitegrid")
    rc("font", family = "serif", size = fontsize)
    font_prop = matplotlib.font_manager.FontProperties(
        family = "serif",
        style = "normal",
        size = fontsize,
    )
    sfmt = matplotlib.ticker.ScalarFormatter(useMathText = true)
    sfmt.set_powerlimits((0, 0))
    matplotlib.use("Agg")

    return font_prop, sfmt
end


function wiggle_plot(
    data::Array{Td,2},
    xrec = nothing,
    time_axis = nothing;
    t_scale = 1.5,
    new_fig = true,
    maximum_x = nothing,
) where {Td}
    # X axis
    if isnothing(xrec)
        @info "No X coordinates prvided, using 1:ntrace"
        xrec = range(0, size(data, 2), length = size(data, 2))
    end
    length(xrec) == size(data, 2) ||
        error("xrec must be the same length as the number of columns in data")
    dx = diff(xrec)
    dx = 2 .* vcat(dx[1], dx)
    # time axis
    if isnothing(time_axis)
        @info "No time axis provided, using 1:ntime"
        time_axis = range(0, size(data, 1), length = size(data, 1))
    end
    length(time_axis) == size(data, 1) ||
        error("time_axis must be the same length as the number of rows in data")
    # Time gain
    tg = time_axis .^ t_scale
    ax = new_fig ? subplots()[2] : gca()

    ax.set_ylim(maximum(time_axis), minimum(time_axis))
    ax.set_xlim(minimum(xrec), maximum(xrec))
    for (i, xr) ∈ enumerate(xrec)
        x = tg .* data[:, i]
        if maximum_x == nothing
            x = dx[i] * x ./ maximum(x) .+ xr
        else
            x = dx[i] * x ./ maximum_x .+ xr
        end
        # rescale to avoid large spikes
        ax.plot(x, time_axis, "k-")
        ax.fill_betweenx(time_axis, xr, x, where = (x .> xr), color = "k")
    end
    ax.set_xlabel("X")
    ax.set_ylabel("Time")
end
