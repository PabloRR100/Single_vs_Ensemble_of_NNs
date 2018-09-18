
from pandas import concat, DataFrame
from collections import OrderedDict as ordict
from beautifultable import BeautifulTable as BT

class TrainResults(object):
    
    def __init__(self, models):
        
        self.m = len(models)
        self.timer = list()
        
        # In case of a single model
        if self.m == 1:
            
            # Store per epoch data
            self.train_loss = list()
            self.train_accy = list()
            
            self.valid_loss = list()
            self.valid_accy = list()
            
            # Store per iteration (training) data
            self.iter_train_loss = list()
            self.iter_train_accy = list()
            
        
        # In case of an ensemble
        else:
            
            # Store per epoch data
            self.train_loss = ordict()
            self.train_accy = ordict()
    
            self.valid_loss = ordict()
            self.valid_accy = ordict()
                        
            # Store per iteration (training) data
            self.iter_train_loss = ordict()
            self.iter_train_accy = ordict()
            
            
            self.train_loss['ensemble'] = list()
            self.train_accy['ensemble'] = list()
            
            self.valid_loss['ensemble'] = list()
            self.valid_accy['ensemble'] = list()
            
            self.iter_train_loss['ensemble'] = list()
            self.iter_train_accy['ensemble'] = list()
                        
            
            for i in range(1, 1 + self.m):
                name = 'm' + str(i)
                self.train_loss[name] = list()
                self.train_accy[name] = list()
                
                self.valid_loss[name] = list()
                self.valid_accy[name] = list()
                
                self.iter_train_loss[name] = list()
                self.iter_train_accy[name] = list()
            
            
    def show(self):
        
        print('Lenght of results collected')
        table = BT()
        table.append_row(['Model', 'Epoch Train', 'Epoch Valid', 'Iter Train'])
        
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
        
        
    def append_time(self, v):
        self.timer.append(v)
        
        
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
        
        # single model
        if self.m == 1: self.iter_train_loss.append(v)             
        # ensemble
        elif not m: self.iter_train_loss['ensemble'].append(v)
        # individual learner
        else: self.iter_train_loss['m' + str(m)].append(v)                        

        
    def append_iter_accy(self, v, subset: str, m=None):
        
        # single model
        if self.m == 1: self.iter_train_accy.append(v)             
        # ensemble
        elif not m: self.iter_train_accy['ensemble'].append(v)
        # individual learner
        else: self.iter_train_accy['m' + str(m)].append(v)  


class TestResults():
    
    def __init__(self):
        
        self.single_accy = None
        self.ensemble_accy = None
        

def aggregateResults(res, eres, test):
    
    # Timer
    timer = concat((DataFrame(res.timer), DataFrame(eres.timer)), axis=1)
    timer.columns = ['Deep model', 'Ensemble']
    
    # Training Loss Per Iteration
    iter_train_loss = concat((DataFrame(res.iter_train_loss, columns=['ResNet56']), 
                                 DataFrame.from_dict(eres.iter_train_loss)), axis=1)
    
    # Training Loss Per Epoch
    epoch_train_loss = concat((DataFrame(res.train_loss, columns=['ResNet56']), 
                                  DataFrame.from_dict(eres.train_loss)), axis=1)
    
    # Training Accuracy Per Iteration
    iter_train_accy = concat((DataFrame(res.iter_train_accy, columns=['ResNet56']), 
                                 DataFrame.from_dict(eres.iter_train_accy)), axis=1)
    
    # Training Accuracy Per Epoch
    epoch_train_accy = concat((DataFrame(res.train_accy, columns=['ResNet56']), 
                                  DataFrame.from_dict(eres.train_accy)), axis=1)
    
    # Training Test Error Per Iteration
    iter_train_testerror = 100 - iter_train_accy.iloc[:,:]
    
    # Training Test Error Per Epoch
    epoch_train_testerror = 100 - epoch_train_accy.iloc[:,:]
    
    # Validation Loss
    valid_loss = concat((DataFrame(res.valid_loss, columns=['ResNet56']), 
                            DataFrame.from_dict(eres.valid_loss)), axis=1)
    
    # Validation Accuracy 
    valid_accy = concat((DataFrame(res.valid_accy, columns=['ResNet56']), 
                            DataFrame.from_dict(eres.valid_accy)), axis=1)
    
    # Validation Test Error
    valid_testerror = 100 - valid_accy.iloc[:,:]
    
    
    # TRAINING DATA
    train = {'iter': 
        
                {'loss': iter_train_loss,
                 'accy': iter_train_accy,
                 'test': iter_train_testerror
                 },
    
             'epoch':
        
                {'loss': epoch_train_loss,
                 'accy': epoch_train_accy,
                 'test': epoch_train_testerror
                 },
       }
    
    
    # VALIDATION DATA
    valid = {'loss': valid_loss,
             'accy': valid_accy,
             'test': valid_testerror
             }
    
    
    
    # TESTING DATA
    test = {'single': test.single_accy,
            'ensemble': test.ensemble_accy}
    
    
    
    # DATA DICTIONARY
    data = {'train': train,
            'valid': valid,
            'test': test,
            'timer': timer}

    return data