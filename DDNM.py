# Define parameter ranges for grid search
k1_range = range(1, 21)  # 1 to 20
k2_range = range(1, 11)  # 1 to 10
qs_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Initialize variables to store best results
best_params = {}
best_accuracy = 0
best_f1 = 0
best_results = {}
all_results = []

# Grid search
for k1 in k1_range:
    for k2 in k2_range:
        for qs in qs_range:
            print(f"\nTesting parameters: k1={k1}, k2={k2}, qs={qs}")

            # Model parameters
            n_step = 5
            iter = 100
            row = train_data.shape[1]
            t = -1
            M = 20

            # Initialize metrics storage for this parameter combination
            err_all = np.zeros((1, n_step))
            Time_all = np.zeros((1, n_step))
            f1_all1 = np.zeros((1, n_step))
            auc_all1 = np.zeros((1, n_step))
            jq_all1 = np.zeros((1, n_step))
            zh_all1 = np.zeros((1, n_step))
            avf1_all1 = np.zeros((1, n_step))
            errr = np.zeros((1, n_step))

            for step in range(n_step):
                # Data preparation for this step
                X = np.vstack((train_data, test_data))
                Y = np.vstack((train_labels, test_labels))
                nall = [i for i in range(len(X))]
                ntrn = len(train_data)
                random.shuffle(nall)
                idxtrn = nall[0:ntrn]
                idxtst = nall[ntrn:]
                train_data = X[idxtrn, :]
                train_labels = Y[idxtrn, :]
                test_data = X[idxtst, :]
                test_labels = Y[idxtst, :]

                x_data = np.vstack((train_data, test_data))
                X_train = train_data
                X_test = test_data
                y_train = train_labels
                y_test = test_labels

                time_start = time.time()
                standardScaler = MinMaxScaler()
                standardScaler.fit(x_data)
                X_train = standardScaler.transform(X_train)
                X_test = standardScaler.transform(X_test)

                # TensorFlow model
                x_orig = tf.compat.v1.placeholder(tf.float32, [None, row])
                y__orig = tf.compat.v1.placeholder(tf.float32, [None, 1])
                W_orig = tf.Variable(tf.compat.v1.random_uniform(shape=[M, row], minval=-1, maxval=1, dtype=tf.float32),
                                     name='W_orig')
                q_orig = tf.Variable(tf.compat.v1.random_uniform(shape=[M, row], minval=-1, maxval=1, dtype=tf.float32),
                                     name='q_orig')

                X_orig = tf.expand_dims(x_orig, 1)
                X_orig = tf.tile(X_orig, [1, M, 1])
                y_temp_orig = tf.nn.sigmoid(k1 * qs * (tf.multiply(X_orig, W_orig) - q_orig))
                y_orig = tf.reduce_prod((y_temp_orig), axis=2)
                y1_orig = (tf.reduce_sum(y_orig, axis=1, keepdims=True))
                y2_orig = tf.nn.sigmoid(k2 * 1.0 * (y1_orig - 0.5))

                loss_orig = tf.reduce_mean(0.5 * tf.square(y__orig - y2_orig))
                train_step_orig = tf.compat.v1.train.AdamOptimizer(0.01).minimize(loss_orig)

                with tf.compat.v1.Session() as sess:
                    sess.run(tf.compat.v1.global_variables_initializer())
                    for _ in range(iter):
                        sess.run(train_step_orig, feed_dict={x_orig: X_train, y__orig: y_train})

                    prediction_value = sess.run(y2_orig, feed_dict={x_orig: X_test})
                    prediction_value = np.int64(prediction_value > 0.5)

                    test_accuracy = accuracy_score(y_test, prediction_value)
                    test_f1 = f1_score(y_test, prediction_value)

                    # Store metrics
                    err_all[0, step] = 1 - test_accuracy
                    Time_all[0, step] = time.time() - time_start

                    f1_all1[0, step] = test_f1
                    fpr1, tpr1, threshold1 = metrics.roc_curve(y_test, prediction_value)
                    roc_auc1 = metrics.auc(fpr1, tpr1)
                    auc_all1[0, step] = roc_auc1
                    jq_all1[0, step] = metrics.precision_score(y_test, prediction_value, average='macro')
                    zh_all1[0, step] = metrics.recall_score(y_test, prediction_value, average='macro')
                    avf1_all1[0, step] = metrics.f1_score(y_test, prediction_value, average='weighted')


