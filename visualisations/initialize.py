import pickle 

class Init:

    def __init__(self, path, lang='ru', d_name='rusenteval'):

        with open(f'{path}scores_{lang}_{d_name}.pkl', 'rb') as f:
            self.scores = pickle.load(f)
    
        with open(f'{path}scores_c_{lang}_{d_name}.pkl', 'rb') as f:
            self.scores_c = pickle.load(f)

        with open(f'{path}scores_layers_{lang}_{d_name}.pkl', 'rb') as f:
            self.scores_layers = pickle.load(f)

        with open(f'{path}neurons_{lang}_{d_name}.pkl', 'rb') as f: # все с отсечками
            self.ordered_neurons = pickle.load(f)
            
        with open(f'{path}top_n_{lang}_{d_name}.pkl', 'rb') as f: #тут 10 проц
            self.top_neurons = pickle.load(f)
            
        with open(f'{path}bottom_n_{lang}_{d_name}.pkl', 'rb') as f: #тут 
            self.bottom_neurons = pickle.load(f)

        with open(f'{path}predicted_{lang}_{d_name}.pkl', 'rb') as f: #тут 
            self.predicted = pickle.load(f) 

        with open(f'{path}threshold_{lang}_{d_name}.pkl', 'rb') as f: # с "трешхолдом"
            self.ordered_neurons_thres = pickle.load(f)
            
        with open(f'{path}scores_keep_bot_{lang}_{d_name}.pkl', 'rb') as f:
            self.scores_keep_bot = pickle.load(f)
            
        with open(f'{path}scores_keep_top_{lang}_{d_name}.pkl', 'rb') as f:
            self.scores_keep_top = pickle.load(f)
            
        with open(f'{path}scores_keep_thres_{lang}_{d_name}.pkl', 'rb') as f:
            self.scores_keep_thres = pickle.load(f)
            
        with open(f'{path}scores_keep_top_c_{lang}_{d_name}.pkl', 'rb') as f:
            self.scores_keep_top_c = pickle.load(f)
            
        with open(f'{path}scores_keep_thres_c_{lang}_{d_name}.pkl', 'rb') as f:
            self.scores_keep_thres_c = pickle.load(f)