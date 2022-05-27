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
        "input"
            help = "the paths al input CSV files"
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
    for input in args["input"]
        df = CSV.read(input, DataFrame)
        title = "\$p_- = $(df[1, :p_minus]), \\, p_+ = $(df[1, :p_plus])\\hphantom{0}\$"

        # collect the plot arguments
        pairs = CriticalDifferenceDiagrams._to_pairs(
            df,
            :method,  # the name of the treatment column
            :dataset, # the name of the observation column
            :f1 # the name of the outcome column
        )

        # sort pairs by their keys
        # pairs = pairs[sortperm(first.(pairs))]
        push!(sequence, title => pairs)
    end

    # generate the 2D critical difference diagram
    plot = CriticalDifferenceDiagrams.plot(
        sequence...;
        maximize_outcome = true,
        alpha = 0.05
    )

    # style it
    plot.style = join([
        plot.style,
        "cycle list={{tu01,mark=*},{tu04,mark=diamond*},{tu02,mark=triangle*},{tu03,mark=square,semithick,densely dotted},{tu05,mark=pentagon,semithick,densely dotted},{tu06,mark=oplus,semithick},{tu07,mark=halfcircle,semithick},{tu04,mark=o},{tu02,mark=diamond, semithick},{tu03,mark=triangle,semithick}}",
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
