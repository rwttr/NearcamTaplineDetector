import Pkg
Pkg.activate(".")
Pkg.instantiate()

""" command line argument parsing section """

using ArgParse

function parse_cmd()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--fold_no"
            help = "specify training datafold"
            arg_type = Int
            default = 0
    end
    return parse_args(s)
end

parsed_args = parse_cmd()
println(typeof(parsed_args))
fold_no = parsed_args["fold_no"]
println(typeof(fold_no))
println("fold_no=$fold_no")
