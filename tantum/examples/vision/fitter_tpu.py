# ====================================================
# main
# ====================================================
def main_fold(fold):
    def get_result(result_df):
        preds = result_df['preds'].values
        labels = result_df[CFG.target_col].values
        score = get_score(labels, preds)
        LOGGER.info(f'Score: {score:<.5f}')
    oof_df = pd.DataFrame()
    fitter = Fitter(
        cfg=CFG,
        model = CustomEfficientNet(model_name=CFG.model_name, pretrained=True),
        device=device,
        optimizer = CFG.optimizer,
        n_epochs = CFG.n_epochs,
        sheduler = CFG.scheduler,
        optimizer_params = CFG.optimizer_params,
        xm=xm,
        pl=pl,
        idist=idist
    )

    trn_idx = folds[folds['fold'] != fold].index
    val_idx = folds[folds['fold'] == fold].index

    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)

    train_dataset = TrainDataset(train_folds, transform=get_transforms(data='train'))
    valid_dataset = TrainDataset(valid_folds, transform=get_transforms(data='valid'))


    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=CFG.num_workers
    )

    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=CFG.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=CFG.num_workers
    )

    _oof_df = fitter.fit(CFG, fold, train_loader, valid_loader, valid_folds)
    oof_df = pd.concat([oof_df, _oof_df])
    
    if CFG.nprocs != 8:
        LOGGER.info(f"========== fold: {fold} result ==========")
        get_result(_oof_df)
                    
        if CFG.nprocs != 8:
            # CV result
            LOGGER.info(f"========== CV ==========")
            get_result(oof_df)
            # save result
            oof_df.to_csv(OUTPUT_DIR+'oof_df.csv', index=False)

if __name__ == '__main__':

    if CFG.device == 'TPU':
        Parallel(n_jobs=5, backend="threading")(delayed(main_fold)(i) for i in range(5))
    elif CFG.device == 'GPU':
        main()