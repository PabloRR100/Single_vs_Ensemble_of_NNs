
from beautifultable import BeautifulTable as BT

class Results(object):
    
    def __init__(self, models):
        
        self.m = len(models)
        
        # Store per epoch training data
        self.train_loss = dict()
        self.train_accy = dict()

        self.valid_loss = dict()
        self.valid_accy = dict()
        
        # Store per iteration training data
        self.global_train_loss = dict()
        self.global_train_accy = dict()
         
        self.global_valid_loss = dict()
        self.global_valid_accy = dict()
        
        # In case of an ensemble
        if self.m > 1:
                        
            for i in range(1, 1 + self.m):
                name = 'm' + str(i)
                self.train_loss[name] = list()
                self.train_accy[name] = list()
                
                self.valid_loss[name] = list()
                self.valid_accy[name] = list()
                
                self.global_train_loss[name] = list()
                self.global_train_accy[name] = list()
                 
                self.global_valid_loss[name] = list()
                self.global_valid_accy[name] = list()
                
            self.train_loss['ensemble'] = list()
            self.train_accy['ensemble'] = list()
            
            self.valid_loss['ensemble'] = list()
            self.valid_accy['ensemble'] = list()
            
            self.global_train_loss['ensemble'] = list()
            self.global_train_accy['ensemble'] = list()
             
            self.global_valid_loss['ensemble'] = list()
            self.global_valid_accy['ensemble'] = list()
            
        # In case of a single model
        else:
            
            self.train_loss = list()
            self.train_accy = list()
            
            self.valid_loss = list()
            self.valid_accy = list()
            
            self.global_train_loss = list()
            self.global_train_accy = list()
             
            self.global_valid_loss = list()
            self.global_valid_accy = list()
            
    def show(self):
        
        print('Lenght of results collected')
        table = BT()
        table.append_row(['Model', 'Per iteration Train', 'Per iteration Valid',
                          'Per epoch Train', 'Per epoch Valid'])
        
        if self.m == 1:
            table.append_row(['Single Deep', len(self.train_loss), len(self.valid_loss),
                              len(self.global_train_loss), len(self.global_valid_loss)])
        
        else:
            for i in range(self.m):
                name = 'm' + str(i+1)
                table.append_row(['Individual {}'.format(i+1), 
                                  len(self.train_loss[name]), len(self.valid_loss[name]),
                                  len(self.global_train_loss[name]), len(self.global_valid_loss[name])])
            
                table.append_row(['Ensemble', 
                                 len(self.train_loss['ensemble']), len(self.valid_loss['ensemble']),
                                 len(self.global_train_loss['ensemble']), len(self.global_valid_loss['ensemble'])])
        print(table)
        
    def append_loss(self, v, subset: str, m=None):
        
        if subset == 'train':
            # ensemble
            if not m: self.train_loss['ensemble'].append(v)
            # single model
            elif self.m == 1: self.train_loss.append(v)             
            # individual learner
            else: self.train_loss['m' + str(m)].append(v)                        
        
        elif subset == 'valid':
            # ensemble
            if not m: self.valid_loss['ensemble'].append(v)
            # single model
            elif self.m == 1: self.valid_loss.append(v)             
            # individual learner
            else: self.valid_loss['m' + str(m)].append(v)            
        
        else: 
            print('Subset must be train or valid!')
            print('Exiting..')
            exit()
        
    def append_accy(self, v, subset: str, m=None):
        
        if subset == 'train':
            # ensemble
            if not m: self.train_accy['ensemble'].append(v)
            # single model
            elif self.m == 1: self.train_accy.append(v)             
            # individual learner
            else: self.train_accy['m' + str(m)].append(v)  

        elif subset == 'valid':
                        # ensemble
            if not m: self.valid_accy['ensemble'].append(v)
            # single model
            elif self.m == 1: self.valid_accy.append(v)             
            # individual learner
            else: self.valid_accy['m' + str(m)].append(v)  
        
        else: 
            print('Subset must be train or valid!')
            print('Exiting..')
            exit()

    def append_global_loss(self, v, subset: str, m=None):
        
        if subset == 'train':
            # ensemble
            if not m: self.train_loss['ensemble'].append(v)
            # single model
            elif self.m == 1: self.train_loss.append(v)             
            # individual learner
            else: self.train_loss['m' + str(m)].append(v)                        
        
        elif subset == 'valid':
            # ensemble
            if not m: self.valid_loss['ensemble'].append(v)
            # single model
            elif self.m == 1: self.valid_loss.append(v)             
            # individual learner
            else: self.valid_loss['m' + str(m)].append(v)            
        
        else: 
            print('Subset must be train or valid!')
            print('Exiting..')
            exit()
        
    def append_global_accy(self, v, subset: str, m=None):
        
        if subset == 'train':
            # ensemble
            if not m: self.global_train_accy['ensemble'].append(v)
            # single model
            elif self.m == 1: self.global_train_accy.append(v)             
            # individual learner
            else: self.global_train_accy['m' + str(m)].append(v)  

        elif subset == 'valid':
                        # ensemble
            if not m: self.global_valid_accy['ensemble'].append(v)
            # single model
            elif self.m == 1: self.global_valid_accy.append(v)             
            # individual learner
            else: self.global_valid_accy['m' + str(m)].append(v)  
        
        else: 
            print('Subset must be train or valid!')
            print('Exiting..')
            exit()
