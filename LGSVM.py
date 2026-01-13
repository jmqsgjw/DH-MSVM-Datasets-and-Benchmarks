"""

Demo example concerning the usage of the LGSVM classifier.

Authors: Marchetti F. & Perracchione E.

"""

# =============================================================================
# DATASET GENERATION
# =============================================================================


data_path="data"

data = scio.loadmat(data_path)
X_train=data['train_data']
y_train=data['train_labels']
X_test=data['test_data']
y_test=data['test_labels']

n_step = 5
err_all = np.zeros((1, n_step))
Time_all = np.zeros((1, n_step))
err1 = np.zeros((n_step, 1))

# =============================================================================
# LGSVM CLASSIFICATION
# =============================================================================
for step in range(0,n_step):
    X = np.vstack((X_train, X_test))
    print('X', X.shape)
    Y = np.vstack((y_train, y_test))
    nall = [i for i in range(len(X))]
    ntrn = len(X_train)
    random.shuffle(nall)
    idxtrn = nall[0:ntrn]
    idxtst = nall[ntrn:]
    X_train = X[idxtrn, :]
    y_train = Y[idxtrn, :]
    X_test = X[idxtst, :]
    y_test = Y[idxtst, :]
    time_start = time.time()
    model = LGSVM(metric = 'cosine', approach = 'sparse', n_centers = 5,
              svm_model = svm.SVC(kernel='linear'))


    model.fit(X_train, y_train.ravel())

    y_predict = model.predict(X_test)
    y_predict = y_predict.reshape((np.size(X_test, 0), 1))
    time_end = time.time()
    err1[step] = np.size(np.where(y_predict + y_test == 0)[1])
    err_all[0, step] = np.sum(err1[step]) / np.size(X_test, axis=0)
    Time_all[0, step] = time_end - time_start


    print('err_all', err_all)
    print('Time_all', Time_all)

print('err_all_mean:', np.mean(err_all))
print('err_all_std:', np.std(err_all))
print('Time_all_sum:', np.sum(Time_all))
print('err_all_max', np.max(err_all))


