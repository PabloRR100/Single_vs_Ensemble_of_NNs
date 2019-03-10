
import os
import pickle


## Introduce the correct path ##
#path = os.path.abspath('../results_II/logs/resnet56.txt')  
#path = os.path.abspath('../results/logs/vggs/vgg13.txt')  
#path = os.path.abspath('../results/logs/vggs/vgg19.txt')  
path = os.path.abspath('../results/logs/resnets/resnet56.txt')  

f = open(path, 'r')
x = f.readlines()[58:-7]
x = [t for t in x if t != '\n']
f.close()

MODELS = 3

def get_loss_accy(tr):
    loss = list(map(float, [t.split(' Loss: ')[1].split(' Acc:')[0] for t in tr]))
    accy = list(map(float, [t.split(' Acc: ')[1].split('%')[0] for t in tr]))
    return loss, accy


# Training
# --------

tr = [t for t in x if 'Train' in t and 'Loss' in t]

# Single Model
tr_single = [t for t in tr if 'Ensemble' not in t]
tr_single_loss, tr_single_accy = get_loss_accy(tr_single)
tr_single_accy_max = max(tr_single_accy)

# Ensemble Model
tr_ensemble = [t for t in tr if 'Ensemble' in t]
tr_ensemble_loss, tr_ensemble_accy = get_loss_accy(tr_ensemble)
tr_ensemble_accy_max = max(tr_ensemble_accy)

# Each learner
## NO INFORMATION RECORDED IN THE LOG FOR THIS



# Validation
# ----------

va = x
va = [t for t in va if 'Valid' in t and 'Loss' in t]

va_single = [t for t in va if 'Ensemble' not in t][:181]
va_single_loss, va_single_accy = get_loss_accy(va_single)

va_singles = []
for i in range(1,MODELS+1):
    va_singles.append([t for t in va if 'Model {}'.format(i) in t])    
va_ensemble = [t for t in va if 'Ensemble' in t]

va_singles_loss = [get_loss_accy(v)[0] for v in va_singles]
va_singles_accy = [get_loss_accy(v)[1] for v in va_singles]
va_ensemble_loss, va_ensemble_accy = get_loss_accy(va_ensemble)

va_single_accy_max = max(va_single_accy)
va_ensemble_accy_max = max(va_ensemble_accy)
va_ensemble_individual_accy_max = max([max(va_ensemble)])



# Testing
# -------

ts = x[-4:]
ts_single = ts[0].split('%')[0][-5:]
ts_ensemble = ts[2].split('%')[0][-5:]



# Timer
# -----

h0, m0 =  x[0].split('Time ')[1].split(':')
h1, m1 = x[640].split('Time ')[1].split(':')
h0, m0, h1, m1 = int(h0), int(m0), int(h1), int(m1)
dh_single = (h1 + 24 - h0 if h1 < h0 else h1 - h0) * 60 + (m1+60-m0 if m1 > m0 else m1)


h0, m0 = x[540].split('Time ')[1].split(':')
h1, m1 = x[1641].split(':')
h0, m0, h1, m1 = int(h0), int(m0), int(h1), int(m1) + 35
dh_essemble = (h1 + 24 - h0 if h1 < h0 else h1 - h0) * 60 + (m1+60-m0 if m1 > m0 else m1)



# Save to Result Class
# --------------------

from results import TrainResults, TestResults

def remove_empty_keys(d):
    d = {k: v for k, v in d.items() if v != []}
    return d


# Single Model

res = TrainResults([0])
res.name = 'ResNet56'
res.train_loss = tr_single_loss
res.train_accy = tr_single_accy
res.valid_loss = va_single_loss
res.valid_accy = va_single_accy
res.timer = dh_single


# Ensemble Model

models = ['m'+str(i) for i in range(1, MODELS+1)]
eres = TrainResults(models)
eres.name = 'ResNet18(x4)'

eres.timer = dh_essemble
eres.train_loss['ensemble'] = tr_ensemble_loss
eres.train_accy['ensemble'] = tr_ensemble_accy
eres.train_loss = remove_empty_keys(eres.train_loss)
eres.train_accy = remove_empty_keys(eres.train_accy)

eres.valid_loss['ensemble'] = va_ensemble_loss
eres.valid_accy['ensemble'] = va_ensemble_accy
for i, m in enumerate(models): 
    eres.valid_loss[m] = va_singles_loss[i]
for i, m in enumerate(models): 
    eres.valid_accy[m] = va_singles_accy[i]


# Testing 

tres = TestResults()
tres.single_accy = ts_single
tres.ensemble_accy = ts_ensemble


with open('Results_Single_Models.pkl', 'wb') as object_result:
    pickle.dump(res, object_result, pickle.HIGHEST_PROTOCOL)

with open('Results_Ensemble_Models.pkl', 'wb') as object_result:
    pickle.dump(eres, object_result, pickle.HIGHEST_PROTOCOL)

with open('Results_Testing.pkl', 'wb') as object_result:
    pickle.dump(tres, object_result, pickle.HIGHEST_PROTOCOL)

