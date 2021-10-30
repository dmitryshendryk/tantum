import numpy as np
import os
from tantum import enums
from tantum.callbacks import Callback


class OutOfFold(Callback):
    def __init__(self, output_path) -> None:
        self.output_path =  output_path

    
    def on_epoch_end(self, model, **kwargs):
        print("Out Of Fold CallBack")

        predictions = model.predictions['valid']['preds']
        valid_folds = model.predictions['valid']['valid_folds']
        valid_folds[[str(c) for c in range(model.target_size)]] = predictions
        valid_folds['preds'] = predictions.argmax(1)
        oof_path = os.path.join(self.output_path, 'oof_df_fold_{}.csv'.format(str(model.fold)))
        valid_folds.to_csv(oof_path, index=False)