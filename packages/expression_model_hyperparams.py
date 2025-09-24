import argparse


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'true', '1'}:
        return True
    elif value.lower() in {'false', '0'}:
        return False
    else:
        raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def script_args(from_command_line=True):
    parser = argparse.ArgumentParser()
    parser.add_argument("--line_selection", choices=["Cux2-Ai96", "Rpb4-Ai96"], default="Cux2-Ai96") #select the functional dataset for tau regressions
    parser.add_argument("--gene_limit", type=int, default=-1) #for testing purposes to load a subset of merfish-imputed data, set to -1 to include all genes
    parser.add_argument("--restrict_merfish_imputed_values", type=str_to_bool, default=False) #condition to restrict merfish-imputed dataset to non-imputed genes
    parser.add_argument("--tau_pool_size_array_full", type=lambda s: [float(item) for item in s.split(',')], default="4.0") #[1,2,3,4,5] #in 25um resolution CCF voxels, converted to mm later
    parser.add_argument("--n_splits", type=int, default=5) #number of splits for cross-validations in regressions
    parser.add_argument("--alpha_params", type=lambda s: [float(item) for item in s.split(',')], default="-5,-2,30") # [Alpha Lower (10**x), Alpha Upper (10**x), Steps]... alpha values for Lasso regressions
    parser.add_argument("--plotting", type=str_to_bool, default=True)
    parser.add_argument("--num_precision", type=int, default=3)   # Just for display (in plotting and regression text files)
    parser.add_argument("--alpha_precision", type=int, default=5) # Just for display (in plotting and regression text files)
    parser.add_argument("--verbose", type=str_to_bool, default=True)    # For print statements
    parser.add_argument("--predictor_order", type=lambda s: [int(item) for item in s.split(',')], default="0")           # Select predictors for regressions, and order [0:merfish{-imputed}, 1:pilot]
    parser.add_argument("--regressions_to_start", type=lambda s: [int(item) for item in s.split(',')], default="0,1,2,3")    # Select predictor/response variables for regressions, and order [0:X->Tau, 1:X->CCF, 2:X,CCF->Tau, 3:X->Tau Res.]
    parser.add_argument("--max_iter", type=int, default=200) # For layer regressions
    parser.add_argument("--variable_management", type=str_to_bool, default=True) # Removes large variables from memory after use (needs to be expanded to include more variables)
    parser.add_argument("--plotting_conditions", type=lambda s: [bool(int(item)) for item in s.split(',')], default="0,1") # For plotting spatial reconstructions
    parser.add_argument("--arg_parse_test", type=str_to_bool, default=False) # For testing the bash argument parser
    parser.add_argument("--job_task_id", type=int, default=0) # For parallel processing
    parser.add_argument("--bootstrapping_scale", type=float, default=1.0) # For CCF coordinate pool bootstrapping (functional-transcriptomic pool registration) scale, default is 1.0
    parser.add_argument("--min_pool_size", type=int, default=3) # Minimum number of pixels and cells in a pool for bootstrapping, default is 3
    parser.add_argument("--preprocessing_only", type=str_to_bool, default=False) # If True, only run preprocessing steps and skip regressions
    parser.add_argument("--output_to_repo", type=str_to_bool, default=True) # If True, output results to the repository

    if not from_command_line:
        # Simulate command-line arguments for testing purposes
        args = parser.parse_args([])
        gene_limit = 2 # For testing purposes, limit the number of genes to 2
    else:
        # Parse the command-line arguments
        args = parser.parse_args()
        gene_limit = args.gene_limit
    
    args_dict = {}
    args_dict['gene_limit'] = gene_limit
    args_dict['line_selection'] = args.line_selection
    args_dict['restrict_merfish_imputed_values'] = args.restrict_merfish_imputed_values
    args_dict['tau_pool_size_array_full'] = args.tau_pool_size_array_full
    args_dict['n_splits'] = args.n_splits
    args_dict['alpha_params'] = args.alpha_params
    args_dict['plotting'] = args.plotting
    args_dict['num_precision'] = args.num_precision
    args_dict['alpha_precision'] = args.alpha_precision
    args_dict['verbose'] = args.verbose
    args_dict['predictor_order'] = args.predictor_order
    args_dict['regressions_to_start'] = args.regressions_to_start
    args_dict['max_iter'] = args.max_iter
    args_dict['variable_management'] = args.variable_management
    args_dict['plotting_conditions'] = args.plotting_conditions
    args_dict['arg_parse_test'] = args.arg_parse_test
    args_dict['job_task_id'] = args.job_task_id
    args_dict['bootstrapping_scale'] = args.bootstrapping_scale
    args_dict['min_pool_size'] = args.min_pool_size
    args_dict['preprocessing_only'] = args.preprocessing_only
    args_dict['output_to_repo'] = args.output_to_repo

    # line_selection = args.line_selection
    # restrict_merfish_imputed_values = args.restrict_merfish_imputed_values
    # tau_pool_size_array_full = args.tau_pool_size_array_full
    # n_splits = args.n_splits
    # alpha_params = args.alpha_params
    # plotting = args.plotting
    # num_precision = args.num_precision
    # alpha_precision = args.alpha_precision
    # verbose = args.verbose
    # predictor_order = args.predictor_order
    # regressions_to_start = args.regressions_to_start
    # max_iter = args.max_iter
    # variable_management = args.variable_management
    # plotting_conditions = args.plotting_conditions
    # arg_parse_test = args.arg_parse_test
    # job_task_id = args.job_task_id
    # bootstrapping_scale = args.bootstrapping_scale
    # min_pool_size = args.min_pool_size
    # preprocessing_only = args.preprocessing_only
    # output_to_repo = args.output_to_repo

    # make sure that alpha_params steps is an integer
    args_dict['alpha_params'][2] = int(args_dict['alpha_params'][2])

    print(f"line_selection: {args_dict['line_selection']}")
    print(f"gene_limit: {args_dict['gene_limit']}")
    print(f"restrict_merfish_imputed_values: {args_dict['restrict_merfish_imputed_values']}")
    print(f"tau_pool_size_array_full: {args_dict['tau_pool_size_array_full']}")
    print(f"n_splits: {args_dict['n_splits']}")
    print(f"alpha_params: {args_dict['alpha_params']}")
    print(f"plotting: {args_dict['plotting']}")
    print(f"num_precision: {args_dict['num_precision']}")
    print(f"alpha_precision: {args_dict['alpha_precision']}")
    print(f"verbose: {args_dict['verbose']}")
    print(f"predictor_order: {args_dict['predictor_order']}")
    print(f"regressions_to_start: {args_dict['regressions_to_start']}")
    print(f"max_iter: {args_dict['max_iter']}")
    print(f"variable_management: {args_dict['variable_management']}")
    print(f"plotting_conditions: {args_dict['plotting_conditions']}")
    print(f"arg_parse_test: {args_dict['arg_parse_test']}")
    print(f"job_task_id: {args_dict['job_task_id']}")
    print(f"bootstrapping_scale: {args_dict['bootstrapping_scale']}")
    print(f"min_pool_size: {args_dict['min_pool_size']}")
    print(f"preprocessing_only: {args_dict['preprocessing_only']}")
    print(f"output_to_repo: {args_dict['output_to_repo']}")

    return args_dict
