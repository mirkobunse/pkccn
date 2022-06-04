using ArgParse, CriticalDifferenceDiagrams, CSV, DataFrames, PGFPlots

# prevent underscores from breaking LaTeX
detokenize(x::Any) = replace(string(x), "_" => "\\_")

"""Parse the command line arguments."""
function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--tex", "-t"
            help = "the path of an output TEX file"
        "--pdf", "-p"
            help = "the path of an output PDF file"
        "--metric", "-m"
            help = "∈ {f1 (default), lima, accuracy}, optionally with prefix \"all_\""
            range_tester = x -> x ∈ ["f1", "lima", "accuracy", "all_f1", "all_lima", "all_accuracy", "DRAFT"]
            default = "f1"
        "--alpha", "-a"
            help = "the alpha value for the hypothesis tests"
            arg_type = Float64
            default = 0.01
        "input"
            help = "the paths of all input CSV files"
            required = true
            nargs = '+'
    end
    return parse_args(s)
end

"""Main function."""
function main(args = parse_commandline())
    @info "Generating plots from $(length(args["input"])) input files" args["tex"] args["pdf"] args["input"] args["alpha"]

    # collect results of all input CSV files
    sequence = Pair{String, Vector{Pair{String, Vector}}}[] # sequence of CD diagrams
    df = DataFrame() # concatenation of all results
    for input in args["input"]
        input_df = CSV.read(input, DataFrame)
        df = vcat(df, input_df)

        # select CDD methods based on the metric
        metric = args["metric"]
        if metric ∈ ["f1", "lima"]
            input_df = input_df[[!occursin("accuracy", x) for x in input_df[!, :method]], :]
        elseif metric == "accuracy"
            input_df = input_df[[!occursin("F1 score", x) for x in input_df[!, :method]], :]
        elseif metric ∈ ["all_f1", "all_lima", "all_accuracy"]
            metric = metric[5:end] # remove the "all_" prefix; keep all methods
        elseif metric == "DRAFT"
            input_df = input_df[[x ∈ [
                "Li \\& Ma threshold (ours; PK-CCN)",
                "Menon et al. (2015; CU-CCN; accuracy)",
                "Mithal et al. (2017; CU-CCN; G measure)"
            ] for x in input_df[!, :method]], :]
            metric = "f1"
        else
            throw(ArgumentError("metric=\"$metric\" is not valid"))
        end

        # collect the plot arguments
        title = "\$p_- = $(input_df[1, :p_minus]), \\, p_+ = $(input_df[1, :p_plus])\$"
        pairs = CriticalDifferenceDiagrams._to_pairs(
            input_df,
            :method,  # the name of the treatment column
            :dataset, # the name of the observation column
            Symbol(metric) # the name of the outcome column
        )
        push!(sequence, title => pairs)
    end

    # # print a table of average scores;
    # # TODO generate a table that focuses the args["metric"]
    # df = df[[x ∈ [
    #     "Li \\& Ma threshold (ours; PK-CCN)",
    #     "Menon et al. (2015; PK-CCN; F1 score)",
    #     "Menon et al. (2015; CU-CCN; F1 score)",
    #     "Menon et al. (2015; CU-CCN; accuracy)",
    #     "Mithal et al. (2017; CU-CCN; G measure)"
    # ] for x ∈ df[!,:method]],:]
    # df[df[!,:method].=="Li \\& Ma threshold (ours; PK-CCN)", :method] .= "Li\\&Ma PK"
    # df[df[!,:method].=="Menon et al. (2015; PK-CCN; F1 score)", :method] .= "Menon PK F1"
    # df[df[!,:method].=="Menon et al. (2015; CU-CCN; F1 score)", :method] .= "Menon CU F1"
    # df[df[!,:method].=="Menon et al. (2015; CU-CCN; accuracy)", :method] .= "Menon CU Acc"
    # df[df[!,:method].=="Mithal et al. (2017; CU-CCN; G measure)", :method] .= "Mithal CU"
    # @info "Tab. 1" agg = unstack( # unstack from long to wide format
    #     vcat( # concatenate noise-wise average and overall average
    #         combine(
    #             groupby(df, [:method, :p_minus, :p_plus]),
    #             Symbol(args["metric"]) => DataFrames.mean => :value
    #         ),
    #         combine(
    #             groupby(df, :method),
    #             Symbol(args["metric"]) => DataFrames.mean => :value,
    #             :p_minus => (x -> -1) => :p_minus, # dummy values
    #             :p_plus => (x -> -1) => :p_plus
    #         )
    #     ),
    #     [:p_minus, :p_plus], # rows
    #     :method, # columns
    #     :value
    # )

    # generate the 2D critical difference diagram
    plot = CriticalDifferenceDiagrams.plot(
        sequence...;
        maximize_outcome = true,
        alpha = args["alpha"]
    )

    # style it
    plot.style = join([
        plot.style,
        "cycle list={{tu01,mark=*},{tu04,mark=diamond*},{tu02,mark=triangle,semithick,densely dotted},{tu03,mark=square,semithick},{tu05,mark=pentagon,semithick},{tu06,mark=oplus,semithick},{tu07,mark=halfcircle,semithick},{tu04,mark=o},{tu02,mark=diamond, semithick},{tu03,mark=triangle,semithick}}",
        "xticklabel style={font=\\small}",
        "yticklabel style={font=\\small}",
        "xlabel style={font=\\small}",
        "legend cell align={left}",
        "width=\\axisdefaultwidth",
        "height=0.9*\\axisdefaultheight"
    ], ", ")
    PGFPlots.resetPGFPlotsPreamble()
    PGFPlots.pushPGFPlotsPreamble(join([
        "\\usepackage{lmodern}",
        "\\definecolor{tu01}{HTML}{84B818}",
        "\\definecolor{tu02}{HTML}{D18B12}",
        "\\definecolor{tu03}{HTML}{1BB5B5}",
        "\\definecolor{tu04}{HTML}{F85A3E}",
        "\\definecolor{tu05}{HTML}{4B6CFC}",
        "\\definecolor{tu06}{HTML}{E3B505}",
        "\\definecolor{tu07}{HTML}{AF331D}"
    ], "\n"))
    if args["pdf"] != nothing
        PGFPlots.save(args["pdf"], plot)
    end
    if args["tex"] != nothing
        PGFPlots.save(args["tex"], plot; limit_to=:picture)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
