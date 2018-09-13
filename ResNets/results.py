
class Results(object):
    
    def __init__(self, models):
        
        self.m = len(models)
        
        self.train_loss = dict()
        self.train_accy = dict()
        
        self.valid_loss = dict()
        self.valid_accy = dict()
        
        if self.m > 1:
                        
            for i in range(1, 1 + self.m):
                name = 'm' + str(i)
                self.train_loss[name] = list()
                self.train_accy[name] = list()
                
                self.valid_loss[name] = list()
                self.valid_accy[name] = list()
                
            self.train_loss['ensemble'] = list()
            self.train_accy['ensemble'] = list()
            
            self.valid_loss['ensemble'] = list()
            self.valid_accy['ensemble'] = list()
        
    def append_loss(self, m, v, subset: str):

        if subset == 'train':
            if m:
                self.train_loss['m' + str(m)].append(v)
            else:
                self.train_loss['ensemble'].append(v)
        
        elif subset == 'valid':
            if m:
                self.valid_loss['m' + str(m)].append(v)
            else:
                self.valid_loss['ensemble'].append(v)
        
        else: 
            print('Subset must be train or valid!')
            print('Exiting..')
            exit()
        
    def append_accy(self, m, v, subset: str):
        
        if subset == 'train':
            if m:
                self.train_accy['m' + str(m)].append(v)
            else:
                self.train_accy['ensemble'].append(v)
        
        elif subset == 'valid':
            if m:
                self.valid_accy['m' + str(m)].append(v)
            else:
                self.valid_accy['ensemble'].append(v)
        
        else: 
            print('Subset must be train or valid!')
            print('Exiting..')
            exit()

    
        
#        
#len(ensemble)
#results = Results(ensemble)