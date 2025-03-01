class ConfigData(object):
    def __init__(self,
        #algorithm = 'MVDLLMAMPlace'
        #algorithm = 'ConvACPlace'
        #algorithm = 'AMPlace'
        #algorithm = 'K_AMPlace'
        #algorithm = 'K_HAAMPlace'
        #algorithm = 'K_GHAAMPlace'
        #algorithm = 'K_IGHAAMPlace'
        algorithm = 'SA',
        #algorithm = 'RLSA'
        #algorithm = 'RLSA2'
        #algorithm = 'RLSA3'
        #algorithm = 'RLSA4'
        #algorithm = 'TwoStep_K_HAAMPlace'
        #algorithm = 'RLsearch_SA'
        #algorithm = 'RLsearch'
        #algorithm = 'MultiStep_RLsearch'
        #algorithm = "DFRegPlace"

        actor_mode = 'normal',
        #actor_mode = 'mv'

        learning_mode = 'training',
        #learning_mode = 'testing'
        #learning_mode = 'donothingwithweights'
        #learning_mode = 'dual'

        greedy_mode = 'hillclimbing',
        #greedy_mode = 'greedy'
        #greedy_mode = 'none',

        #save_all_state = True
        save_all_state = False,

        test_mode='normal',
        #test_mode='random'

        #initial placement type
        input_placement_type = '2d',

        #parameter to hanle sqlite Max_columns limitation
        columns_mod = 1,

        integration_mode='2d',
        single_cell_height = 9,
        layer_values = [ 0, 1 ],
        bin_size_x = 4,
        bin_size_y = 1,
        import_num_sites = 32,
        import_num_rows = 32,

        state_method = 'seq',
        #state_method = 'onehot'

        #sequence_type = 'single'
        #sequence_type = 'triple'
        sequence_type = 'enhancedTriple',

        action_method = 'onehot',
        #action_method = 'nodescount'

        folding_action_method = 'parametric',

        #cost_function = 'twl'
        cost_function = 'mixed'):


        self.algorithm = algorithm
        self.actor_mode = actor_mode
        self.learning_mode = learning_mode
        self.greedy_mode = greedy_mode
        self.save_all_state = save_all_state
        self.test_mode = test_mode
        self.input_placement_type = input_placement_type
        self.columns_mod = columns_mod
        self.integration_mode = integration_mode
        self.single_cell_height = single_cell_height
        self.layer_values = layer_values
        self.bin_size_x = bin_size_x
        self.bin_size_y = bin_size_y
        self.import_num_sites = import_num_sites
        self.import_num_rows = import_num_rows
        self.state_method = state_method
        self.sequence_type = sequence_type
        self.action_method = action_method
        self.folding_action_method = folding_action_method
        self.cost_function = cost_function

        if (self.import_num_sites != self.import_num_rows):
            raise ValueError("Only square layouts are supported")
