import numpy as np
import scipy.io as scio
import h5py
import time
import random
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load data
data_path = "data.mat"
data = scio.loadmat(data_path)
train_data = data['train_data']
train_labels = data['train_labels'].ravel()
test_data = data['test_data']
test_labels = data['test_labels'].ravel()


# ==============Main adjustment parameters==============
n_step = 1  # Number of experiments
TT = 20  # Number of distributed clients
kernel = 'linear'  # Kernel type: ‘linear’, ‘rbf’, 'poly'

N = len(train_data) // TT
TDATA = train_data.copy()
TLABEL = train_labels.copy()
total_time = 0
error_rates = []

# =================================Start experiment==============================
for step in range(n_step):
    X = np.vstack((TDATA, test_data))
    Y = np.hstack((TLABEL, test_labels))

    nall = [i for i in range(len(X))]
    ntrn = len(TDATA)
    random.shuffle(nall)
    idxtrn = nall[0:ntrn]
    idxtst = nall[ntrn:]

    train_data = X[idxtrn, :]
    train_labels = Y[idxtrn]
    test_data = X[idxtst, :]
    test_labels = Y[idxtst]

    AA = train_data[0:TT * N]
    A = np.array_split(AA, TT)

    BB = train_labels[0:TT * N]
    B = np.array_split(BB, TT)

    print("\n Number of training sessions=============:", step + 1)
    print("Start basic SVM training=============")
    time_start_iid = time.time()
    svm_model = SVC(kernel=kernel)

    for kk in range(TT - 1):
        train_data_block = A[kk]
        train_labels_block = B[kk]
        print("Start training the client：", kk + 1)
        svm_model.fit(train_data_block, train_labels_block)

    train_data_final = train_data[((TT - 1) * N):, :]
    train_labels_final = train_labels[((TT - 1) * N):]
    print("Start training the client：", TT)
    svm_model.fit(train_data_final, train_labels_final)

    time_end_iid = time.time()
    step_time = time_end_iid - time_start_iid
    total_time += step_time
    print('Training time for this session: %.4fs' % step_time)

    y_pred = svm_model.predict(test_data)
    accuracy = accuracy_score(test_labels, y_pred)
    error_rate = 1 - accuracy
    error_rates.append(error_rate)
    print('The error rate for this assessment: %.4f' % error_rate)

# Calculate statistical results
avg_error_rate = np.mean(error_rates)
std_error_rate = np.std(error_rates)

# Output final statistical results
print("\n=============Final statistics results=============")
print(f"Total number of repeated tests: {n_step}")
print(f"Total training time: {total_time:.4f}s")
print(f"Average error rate: {avg_error_rate:.4f}")
print(f"Standard deviation of error rate: {std_error_rate:.4f}")