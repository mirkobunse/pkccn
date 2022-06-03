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
            help = "the metric to consider ∈ {f1 (default), lima, accuracy}"
            range_tester = x -> x ∈ ["f1", "lima", "accuracy"]
            default = "f1"
        "--alpha", "-a"
            help = "the alpha value for the hypothesis tests"
            default = 0.1
        "input"
            help = "the paths of all input CSV files"
            required = true
            nargs = '+'
    end
    return parse_args(s)
end

"""Main function."""
function main(args = parse_commandline())
    @info "Generating plots from $(length(args["input"])) input files" args["tex"] args["pdf"] args["input"]

    # collect results of all input CSV files
    sequence = Pair{String, Vector{Pair{String, Vector}}}[] # sequence of CD diagrams
    df = DataFrame() # concatenation of all results
    for input in args["input"]
        input_df = CSV.read(input, DataFrame)
        df = vcat(df, input_df)

        # collect the plot arguments
        title = "\$p_- = $(input_df[1, :p_minus]), \\, p_+ = $(input_df[1, :p_plus])\$"
        pairs = CriticalDifferenceDiagrams._to_pairs(
            input_df,
            :method,  # the name of the treatment column
            :dataset, # the name of the observation column
            Symbol(args["metric"]) # the name of the outcome column
        )
        push!(sequence, title => pairs)
    end

    # print a table of average scores
    df[df[!,:method].=="Li \\& Ma threshold (ours; PK-CCN)", :method] .= "Li\\&Ma PK"
    df[df[!,:method].=="Menon et al. (2015; PK-CCN; accuracy)", :method] .= "Menon PK"
    df[df[!,:method].=="Menon et al. (2015; CK-CCN; accuracy)", :method] .= "Menon CK"
    df[df[!,:method].=="Menon et al. (2015; CU-CCN; accuracy)", :method] .= "Menon CU"
    df[df[!,:method].=="Mithal et al. (2017; CU-CCN; G measure)", :method] .= "Mithal CU"
    df[df[!,:method].=="default (accuracy)", :method] .= "default"
    @show "Tab. 1" agg = unstack( # unstack from long to wide format
        vcat( # concatenate noise-wise average and overall average
            combine(
                groupby(df, [:method, :p_minus, :p_plus]),
                Symbol(args["metric"]) => DataFrames.mean => :value
            ),
            combine(
                groupby(df, :method),
                Symbol(args["metric"]) => DataFrames.mean => :value,
                :p_minus => (x -> -1) => :p_minus, # dummy values
                :p_plus => (x -> -1) => :p_plus
            )
        ),
        [:p_minus, :p_plus], # rows
        :method, # columns
        :value
    )

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
