import json
import os

"""
This module will generate the model parameters
that are needed for training of the model.
"""
cats = (
    'plane', 'car', 'bench', 'chair', 'sofa',  'table', 'cabinet', 'monitor',
    'lamp', 'speaker', 'watercraft', 'cellphone', 'pistol')
param_types = ('base', 'entropy', 'log-weighted', 'regularized')

params = {
    'base': {},
    'entropy': {
        'entropy_loss': {
                'weight': 1e2,
                'exp_annealing_rate': 1e-4
        }
    },
    'log-weighted': {
        'gamma': 'log',
        'prob_eps': 1e-3
    },
    'regularized': {
        'dp_regularization': {
            'weight': 1e0,
            'exp_annealing_rate': 1e-4
        }
    },
    'rm1': {
        'dp_regularization': {
            'weight': 1e-1,
            'exp_annealing_rate': 1e-4
        }
    }
}
for k, v in params.items():
    v['inference_params'] = {'alpha': 0.25}

for cat in cats:
    if not os.path.exists(cat):
        os.mkdir(cat)
    for p in param_types:
        v = params[p]
        v['cat_desc'] = cat
        v['learning_rate'] = 1e-3
        path = cat + os.sep + ('%s_%s.json' % (p, cat))

        with open(path, 'w') as fp:
            json.dump(v, fp)
print("Model parameters are generated")
