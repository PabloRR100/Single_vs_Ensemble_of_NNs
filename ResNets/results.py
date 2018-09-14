
from beautifultable import BeautifulTable as BT

class Results(object):
    
    def __init__(self, models):
        
        self.m = len(models)
        
        # In case of a single model
        if self.m == 1:
            
            # Store per epoch training data
            self.train_loss = list()
            self.train_accy = list()
            
            self.valid_loss = list()
            self.valid_accy = list()
            
            # Store per iteration training data
            self.iter_train_loss = list()
            self.iter_train_accy = list()
        
        # In case of an ensemble
        else:
            
            # Store per epoch training data
            self.train_loss = dict()
            self.train_accy = dict()
    
            self.valid_loss = dict()
            self.valid_accy = dict()
            
            # Store per iteration training data
            self.iter_train_loss = dict()
            self.iter_train_accy = dict()
                        
            for i in range(1, 1 + self.m):
                name = 'm' + str(i)
                self.train_loss[name] = list()
                self.train_accy[name] = list()
                
                self.valid_loss[name] = list()
                self.valid_accy[name] = list()
                
                self.iter_train_loss[name] = list()
                self.iter_train_accy[name] = list()
                
            self.train_loss['ensemble'] = list()
            self.train_accy['ensemble'] = list()
            
            self.valid_loss['ensemble'] = list()
            self.valid_accy['ensemble'] = list()
            
            self.iter_train_loss['ensemble'] = list()
            self.iter_train_accy['ensemble'] = list()
            
            
    def show(self):
        
        print('Lenght of results collected')
        table = BT()
        table.append_row(['Model', 'Per iteration Train',
                          'Per epoch Train', 'Per epoch Valid'])
        
        if self.m == 1:
            table.append_row(['Single Deep', len(self.train_loss), 
                              len(self.valid_loss), len(self.iter_train_loss)])
        
        else:
            for i in range(self.m):
                name = 'm' + str(i+1)
                table.append_row(['Individual {}'.format(i+1), len(self.train_loss[name]), 
                                  len(self.valid_loss[name]), len(self.iter_train_loss[name])])
            
            table.append_row(['Ensemble', len(self.train_loss['ensemble']), 
                             len(self.valid_loss['ensemble']), len(self.iter_train_loss['ensemble'])])
        print(table)
        
    def append_loss(self, v, subset: str, m=None):
        
        if subset == 'train':
            # single model
            if self.m == 1: self.train_loss.append(v)             
            # ensemble
            elif not m: self.train_loss['ensemble'].append(v)
            # individual learner
            else: self.train_loss['m' + str(m)].append(v)                        
        
        elif subset == 'valid':
            # single model
            if self.m == 1: self.valid_loss.append(v)             
            # ensemble
            elif not m: self.valid_loss['ensemble'].append(v)
            # individual learner
            else: self.valid_loss['m' + str(m)].append(v)            
        
        else: 
            print('Subset must be train or valid!')
            print('Exiting..')
            exit()
        
    def append_accy(self, v, subset: str, m=None):
        
        if subset == 'train':
            # single model
            if self.m == 1: self.train_accy.append(v)             
            # ensemble
            elif not m: self.train_accy['ensemble'].append(v)
            # individual learner
            else: self.train_accy['m' + str(m)].append(v)  

        elif subset == 'valid':
            # single model
            if self.m == 1: self.valid_accy.append(v)             
            # ensemble
            elif not m: self.valid_accy['ensemble'].append(v)
            # individual learner
            else: self.valid_accy['m' + str(m)].append(v)  
        
        else: 
            print('Subset must be train or valid!')
            print('Exiting..')
            exit()

    def append_iter_loss(self, v, subset: str, m=None):
        
        if subset == 'train':
            # single model
            if self.m == 1: self.train_loss.append(v)             
            # ensemble
            elif not m: self.train_loss['ensemble'].append(v)
            # individual learner
            else: self.train_loss['m' + str(m)].append(v)                        
        
        elif subset == 'valid':
            # single model
            if self.m == 1: self.valid_loss.append(v)             
            # ensemble
            elif not m: self.valid_loss['ensemble'].append(v)
            # individual learner
            else: self.valid_loss['m' + str(m)].append(v)            
        
        else: 
            print('Subset must be train or valid!')
            print('Exiting..')
            exit()
        
    def append_iter_accy(self, v, subset: str, m=None):
        
        if subset == 'train':
            # single model
            if self.m == 1: self.iter_train_accy.append(v)             
            # ensemble
            elif not m: self.iter_train_accy['ensemble'].append(v)
            # individual learner
            else: self.iter_train_accy['m' + str(m)].append(v)  

        elif subset == 'valid':
            # single model
            if self.m == 1: self.iter_valid_accy.append(v)             
            # ensemble
            elif not m: self.iter_valid_accy['ensemble'].append(v)
            # individual learner
            else: self.iter_valid_accy['m' + str(m)].append(v)  
        
        else: 
            print('Subset must be train or valid!')
            print('Exiting..')
            exit()
