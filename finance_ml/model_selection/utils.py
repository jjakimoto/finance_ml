def get_train_times(t1, test_times):
    trn = t1.copy(deep=True)
    for i, j in test_times.iteritems():
        df0 = trn[(i <= trn.index) & (trn.index <= j)].index
        df1 = trn[(i <= trn) & (trn <= j)].index
        df2 = trn[(trn.index <= i) & (j <= trn)].index
        trn = trn.drop(df0.union(df1.union(df2)))
    return trn
