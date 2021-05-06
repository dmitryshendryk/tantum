import numpy as np
from tqdm import tqdm
from itertools import combinations, chain
from sklearn.metrics import accuracy_score




def get_best_ensemble(df, columns, top=50):
    combined = []
    for i in range(len(columns)):
        combined.append(list(combinations(columns, i+1)))

    def evaluate_ensemble(df, columns):
        return df[[*columns]].apply(lambda x: np.argmax([np.sum(v) for v in zip(*[x[c] for c in columns])]), axis=1).values

    results = dict()
    with tqdm(total=len(list(chain(*combined)))) as process_bar:
        for c in list(chain(*combined)):
            process_bar.update(1)  
            results[c] = accuracy_score(df.label.values, evaluate_ensemble(df, c))
    
    return {k: results[k] for k in sorted(results, key=results.get, reverse=True)[0:top]}