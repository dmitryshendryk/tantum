import numpy as np
import os

import torch


class ClassificationTTaWrapper():
    def __init__(self, fitter, target_size, transforms) -> None:

        self.transforms = transforms
        self.target_size = target_size
        self.fitter = fitter


    def predict(self, fold, dataset, tta=5, device='cpu', apply_tta = False):
        print('=' * 20, 'Fold', fold, '=' * 20)

        self.fitter.model.to(device)

        # PREDICT
        print('Predicting...')
        if apply_tta:
            predictions = np.zeros([len(dataset),self.target_size])
            
            for i in range(tta): 
                tta_predictions = self.fitter.predict(dataset)
                tta_predictions = np.vstack(tta_predictions)
                predictions += tta_predictions/tta  
            predictions = predictions.reshape((len(dataset),1, self.target_size))    
        else:
            predictions = self.fitter.predict(dataset)

        return predictions
    
    def predict_folds(self, 
                      submission_df,
                      models_path,
                      n_folds, 
                      dataset, 
                      tta=5, 
                      device='cpu', 
                      apply_tta = False):

        final_preds = None
        models_path.sort()
        
        for i in range(n_folds):
            self.fitter.load_state_dict(torch.load(models_path[i],
                                 map_location=torch.device(device))['state_dict'])

            preds = self.predict(fold = i, dataset=dataset, tta=tta, device=device, apply_tta=apply_tta)
            temp_preds = None
            for p in preds:
                if temp_preds is None:
                    temp_preds = p
                else:
                    temp_preds = np.vstack((temp_preds, p))
            if final_preds is None:
                final_preds = temp_preds
            else:
                final_preds += temp_preds

        final_preds /= n_folds
        final_preds = final_preds.argmax(axis=1)
 
        submission_df.label = final_preds
        
        return submission_df