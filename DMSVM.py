def svm_class(X, Y, C, tol, kernel, gamma):
    clf = SVC(C=C, tol=tol, kernel=kernel, gamma=gamma)
    clf.fit(X, Y)
    return clf

def svm_kernel(X, kernel, gamma, X_sup=None):
    if kernel == 'rbf':
        if X_sup is None:
            return np.exp(-gamma * np.sum(X**2, axis=1))
        else:
            return np.exp(-gamma * np.sum((X - X_sup)**2, axis=1))
    else:
        raise ValueError("Unsupported kernel")

n_step = 5
mtimes = 1
kn = 10
kernel = 'rbf'
C = 100
kerneloption = 1


ntrn = train_data.shape[0]
n = int(ntrn / kn)
Mi = 2000
Mm = 2000
tiid = 0
tm = 0

stepm = 1
while stepm <= mtimes:
    NS = Mm
    i = 0
    XtrainM[i] = XX_train[a]
    YtrainM[i] = YY_train[a]
    IdxM[i] = a

    K11 = svm_kernel(XtrainM[i: i +1], kernel, kerneloption, clf1.support_vectors_)
    yy = K11.dot(clf1.dual_coef_.T) + clf1.intercept_
    p1 = np.exp(-max(1 - yy * YtrainM[i], 0))
    q1 = np.exp(-yy * YtrainM[i])
    increase = 1
    count = 0

    while i < NS - 1:
        afi = np.where(a == IdxM[: i +1])[0]
        if len(afi) == 0:
            K12 = svm_kernel(XX_train[a: a +1], kernel, kerneloption, clf1.support_vectors_)
            y2 = K12.dot(clf1.dual_coef_.T) + clf1.intercept_
            p2 = np.exp(-max(1 - y2 * YY_train[a], 0))
            q2 = np.exp(-y2 * YY_train[a])
            r = min(1, p2 / p1)

            if r == 1:
                if (YY_train[a] == -1) and (YtrainM[i] == -1):
                    r = q2 / q1
                if (YY_train[a] == 1) and (YtrainM[i] == 1):
                    r = q1 / q2

            probabi = np.random.rand()
            if probabi > r:
                if count >= 5:
                    increase *= 1.3
                    r = increase * r
                count += 1

            if (probabi <= r) or (count > 10):
                i += 1
                IdxM[i] = a
                XtrainM[i] = XX_train[a]
                YtrainM[i] = YY_train[a]
                p1 = p2
                q1 = q2
                increase = 1
                count = 0

        a = np.random.randint(XX_train.shape[0])

    clf2 = svm_class(XtrainM[: i +1], YtrainM[: i +1], C, 1e-7, kernel, kerneloption)
    K2 = svm_kernel(XX_train, kernel, kerneloption, clf2.support_vectors_)
    ytrain = np.sign(K2.dot(clf2.dual_coef_.T) + clf2.intercept_)
    stepm += 1
    e = np.mean(YY_train != ytrain.flatten())

    if e > 0.5:
        stepm -= 1

alm = 0.5 * np.log((1 - e) / e) if e != 0 else 1e6
x += alm
tm += time.time() - start

K3 = svm_kernel(test_data, kernel, kerneloption, clf2.support_vectors_)
ymtest += (K3.dot(clf2.dual_coef_.T) + clf2.intercept_) * alm

ymtest /= x

tp = fn = fp = tpm = fnm = fpm = 0

yiid_pred = np.sign(yiidtest)

err_yiid[step] = accuracy_score(test_labels, yiid_pred)
