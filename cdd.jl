using ArgParse, CriticalDifferenceDiagrams, CSV, DataFrames, Printf, PGFPlots

# prevent underscores from breaking LaTeX
detokenize(x::Any) = replace(string(x), "_" => "\\_")

"""Parse the command line arguments."""
function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--tex", "-t"
            help = "the path of an output TEX file for a CD diagram"
        "--pdf", "-p"
            help = "the path of an output PDF file for a CD diagram"
        "--average-table"
            help = "the path of an output TEX file for an average score table"
        "--dataset-table"
            help = "the path of an output TEX file for a dataset-wise score table"
        "--metric", "-m"
            help = "∈ {f1 (default), lima, accuracy}, optionally with prefix \"all_\""
            range_tester = x -> x ∈ ["f1", "lima", "accuracy", "all_f1", "all_lima", "all_accuracy", "DRAFT"]
            default = "f1"
        "--alpha", "-a"
            help = "the alpha value for the hypothesis tests"
            arg_type = Float64
            default = 0.05
        "input"
            help = "the paths of all input CSV files"
            required = true
            nargs = '+'
    end
    return parse_args(s)
end

abbreviate_method_name(x) = # abbreviate the name for a table of average scores
    if x == "Li \\& Ma threshold (ours; PK-CCN)"
        "Li\\&Ma\\\\thresh."
    elseif x == "Li \\& Ma tree (ours; PK-CCN)"
        "Li\\&Ma\\\\tree"
    elseif x == "Menon et al. (2015; PK-CCN; accuracy)"
        "Menon\\\\PK/acc."
    elseif x == "Menon et al. (2015; PK-CCN; F1 score)"
        "Menon\\\\PK/F1"
    elseif x == "Menon et al. (2015; CK-CCN; accuracy)"
        "Menon\\\\CK/acc."
    elseif x == "Menon et al. (2015; CK-CCN; F1 score)"
        "Menon\\\\CK/F1"
    elseif x == "Menon et al. (2015; CU-CCN; accuracy)"
        "Menon\\\\CU/acc."
    elseif x == "Menon et al. (2015; CU-CCN; F1 score)"
        "Menon\\\\CU/F1"
    elseif x == "Mithal et al. (2017; CU-CCN; G measure)"
        "Mithal\\\\CU/G"
    elseif x == "Yao et al. (2020; CU-CCN; accuracy)"
        "Yao\\\\CU/acc."
    elseif x == "default (accuracy)"
        "default\\\\acc."
    elseif x == "default (F1 score)"
        "default\\\\F1"
    else
        @warn "No abbreviated for the method name \"$(x)\""
        x # return x without abbreviating
    end

format_f1(x) = try @sprintf("%.3f", x)[2:end] catch; "" end
format_lima(x) = try @sprintf("%.2f", x) catch; "" end

function save_table(output_path, df)
    @info "Exporting a LaTeX table to $(output_path)"
    open(output_path, "w") do io
        println(io, "\\begin{tabular}{l$(repeat("c", size(df, 2)-1))}")
        println(io, "  \\toprule")
        println(io, "    ", join(.*(
            "\\makecell{",
            names(df),
            "}"
        ), " & "), " \\\\") # header
        println(io, "  \\midrule")
        for r in eachrow(df)
            println(io, "    ", join(r, " & "), " \\\\")
        end
        println(io, "  \\bottomrule")
        println(io, "\\end{tabular}")
    end
    return nothing
end

"""Main function."""
function main(args = parse_commandline())
    @info "Generating plots from $(length(args["input"])) input files" args["tex"] args["pdf"] args["input"] args["alpha"]

    # collect the results of all input CSV files
    df = DataFrame()
    for input in args["input"]
        df = vcat(df, CSV.read(input, DataFrame), cols=:union)
    end

    # select CDD methods based on the desired metric
    metric = args["metric"]
    styles = [
        "{tu01,mark=*}", # Li & Ma threshold
        "{tu02,mark=diamond*}", # Menon PK-CCN
        "{tu02,mark=diamond,semithick,densely dotted}", # Menon CK-CCN
        "{tu02,mark=diamond,semithick}", # Menon CU-CCN
        "{tu05,mark=pentagon,semithick}", # Mithal
        "{tu04,mark=square,semithick}", # default
        "{tu04,mark=o}",
        "{tu02,mark=diamond, semithick}",
        "{tu03,mark=triangle,semithick}"
    ]
    if metric ∈ ["f1", "lima"]
        df = df[[!occursin("accuracy", x) for x in df[!, :method]], :]
    elseif metric == "accuracy"
        df = df[[!occursin("F1 score", x) for x in df[!, :method]], :]
    elseif metric ∈ ["all_f1", "all_lima", "all_accuracy"]
        metric = metric[5:end] # remove the "all_" prefix; keep all methods
    elseif metric == "DRAFT"
        df = df[[x ∈ [
            "Li \\& Ma threshold (ours; PK-CCN)",
            "Menon et al. (2015; CU-CCN; accuracy)",
            "Mithal et al. (2017; CU-CCN; G measure)",
            "default (accuracy)", # ≡ Yao
            "Li \\& Ma tree (ours; PK-CCN)",
        ] for x in df[!, :method]], :]
        metric = "f1"
        styles = styles[[1, 4, 5, 6, 7]]
    else
        throw(ArgumentError("metric=\"$metric\" is not valid"))
    end

    # print a table of average scores (Tab. 1)
    format_score = if args["metric"] == "lima" format_lima else format_f1 end
    df[!,:abbreviation] = abbreviate_method_name.(df[!,:method])
    df = df[df[!,:abbreviation] .!= "Li\\&Ma\\\\tree", :] # ignore the LiMa tree
    average_scores = vcat(
        transform(
            groupby(vcat( # concatenate noise-wise average and overall average
                combine(
                    groupby(df[df[!,:abbreviation] .!= "Menon\\\\CK/F1", :], [:abbreviation, :p_minus, :p_plus]),
                    Symbol(args["metric"]) => DataFrames.mean => :mean,
                    Symbol(args["metric"]*"_std") => DataFrames.mean => :std
                ),
                combine(
                    groupby(df[df[!,:abbreviation] .!= "Menon\\\\CK/F1", :], :abbreviation),
                    Symbol(args["metric"]) => DataFrames.mean => :mean,
                    Symbol(args["metric"]*"_std") => DataFrames.mean => :std,
                    :p_minus => (x -> -1) => :p_minus, # dummy values
                    :p_plus => (x -> -1) => :p_plus
                )
            ), [:p_minus, :p_plus]),
            :mean => (x -> x .== maximum(x)) => :is_best
        ),
        transform(
            groupby(vcat( # concatenate noise-wise average and overall average
                combine(
                    groupby(df[df[!,:abbreviation] .== "Menon\\\\CK/F1", :], [:abbreviation, :p_minus, :p_plus]),
                    Symbol(args["metric"]) => DataFrames.mean => :mean,
                    Symbol(args["metric"]*"_std") => DataFrames.mean => :std
                ),
                combine(
                    groupby(df[df[!,:abbreviation] .== "Menon\\\\CK/F1", :], :abbreviation),
                    Symbol(args["metric"]) => DataFrames.mean => :mean,
                    Symbol(args["metric"]*"_std") => DataFrames.mean => :std,
                    :p_minus => (x -> -1) => :p_minus, # dummy values
                    :p_plus => (x -> -1) => :p_plus
                )
            ), [:p_minus, :p_plus]),
            :mean => (x -> false) => :is_best
        )
    )

    average_scores[!,:mean_fmt] = format_score.(average_scores[!,:mean])
    average_scores[!,:std_fmt] = format_score.(average_scores[!,:std])
    average_scores[!,:value] = .*(
        [x ? "\$\\mathbf{" : "\${" for x ∈ average_scores[!,:is_best]],
        [x ? "\\hphantom{0}" : "" for x ∈ ((average_scores[!,:mean].<10).&(args["metric"]=="lima"))],
        average_scores[!,:mean_fmt],
        "\\pm",
        average_scores[!,:std_fmt],
        "}\$"
    ) # element-wise concatenation of strings
    average_scores[!,Symbol("\$(p_-, p_+)\$")] = .*(
        "\$(",
        string.(average_scores[!,:p_minus]),
        ", ",
        string.(average_scores[!,:p_plus]),
        ")\$",
    )
    average_scores = unstack( # unstack from long to wide format
        average_scores,
        Symbol("\$(p_-, p_+)\$"), # rows
        :abbreviation, # columns
        :value
    )
    average_scores[end,Symbol("\$(p_-, p_+)\$")] = "avg."
    if args["average-table"] != nothing
        save_table(args["average-table"], average_scores)
    end

    # print a second table without aggregating over data sets (appendix)
    dataset_scores = transform(
        groupby(combine(
            groupby(df, [:abbreviation, :dataset, :p_minus, :p_plus]),
            Symbol(args["metric"]) => DataFrames.mean => :mean,
            Symbol(args["metric"]*"_std") => DataFrames.mean => :std
        ), [:dataset, :p_minus, :p_plus]),
        :mean => (x -> x .== maximum(x)) => :is_best
    )
    dataset_scores[!,:mean_fmt] = format_score.(dataset_scores[!,:mean])
    dataset_scores[!,:std_fmt] = format_score.(dataset_scores[!,:std])
    dataset_scores[!,:value] = .*(
        [x ? "\$\\mathbf{" : "\${" for x ∈ dataset_scores[!,:is_best]],
        dataset_scores[!,:mean_fmt],
        "\\pm",
        dataset_scores[!,:std_fmt],
        "}\$"
    )
    dataset_scores = sort(unstack(
        dataset_scores,
        [:dataset, :p_minus, :p_plus], # rows
        :abbreviation, # columns
        :value
    ), :dataset)
    dataset_scores[!,:dataset] = convert.(String, dataset_scores[!,:dataset]) # variable-width Strings
    last_dataset = ""
    for r in eachrow(dataset_scores)
        dataset = replace(r[:dataset], "_"=>"-")
        if dataset != last_dataset
            r[:dataset] = "\\multirow{6}{*}{\\rotatebox[origin=c]{90}{\\tiny $(dataset)}}"
            last_dataset = dataset
        else
            r[:dataset] = ""
        end
    end
    rename!(dataset_scores, "p_minus" => "\$p_-\$")
    rename!(dataset_scores, "p_plus" => "\$p_+\$")
    rename!(dataset_scores, "dataset" => "")
    if args["dataset-table"] != nothing
        save_table(args["dataset-table"], dataset_scores)
    end

    # generate a sequence of CD diagrams
    sequence = Pair{String, Vector{Pair{String, Vector}}}[]
    for ((p_minus, p_plus), gdf) in pairs(groupby(df, [:p_minus, :p_plus]))
        if length(string(p_plus)) == 3
            p_plus = "$(p_plus)\\hphantom{0}" # left-align tick marks
        end
        title = "\$p_- = $(p_minus), \\, p_+ = $(p_plus)\$"
        pairs = CriticalDifferenceDiagrams.to_pairs(
            gdf,
            :method,  # the name of the treatment column
            :dataset, # the name of the observation column
            Symbol(metric) # the name of the outcome column
        )
        push!(sequence, title => pairs)
    end
    plot = CriticalDifferenceDiagrams.plot( # generate the 2D diagram
        sequence...;
        maximize_outcome = true,
        adjustment = :bonferroni,
        alpha = args["alpha"]
    )

    # style it
    plot.style = join([
        plot.style,
        "cycle list={$(join(styles, ','))}",
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
