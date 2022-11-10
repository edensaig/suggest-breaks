import surprise

__all__=[
    'stratified_train_test_split',
]

def stratified_train_test_split(dataset, train_size, random_state=None):
    """
    Randomly split the surprise ratings dataset to train and test, stratified by user id.
    """
    user_ratings = {}
    rng = surprise.utils.get_rng(random_state)
    for r in dataset.raw_ratings:
        uid = r[0]
        if uid not in user_ratings:
            user_ratings[uid] = list()
        user_ratings[uid].append(r)
    raw_trainset = []
    raw_testset = []
    for uid in user_ratings:
        r = user_ratings[uid]
        rng.shuffle(r)
        n_train = int(round(len(r)*train_size))
        raw_trainset.extend(r[:n_train])
        raw_testset.extend(r[n_train:])
    trainset = dataset.construct_trainset(raw_trainset)
    testset = dataset.construct_testset(raw_testset)
    return trainset, testset
