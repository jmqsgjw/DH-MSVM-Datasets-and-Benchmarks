def Fisher(train_features, train_targets,test_features,test_targets):
 train_pos = [i for i in range(len(train_targets)) if train_targets[i]==1]
 train_neg = [i for i in range(len(train_targets)) if train_targets[i]==-1]

 s0 = np.cov(train_features[train_neg,:].T,ddof =0)
 m0 = np.mean(train_features[train_neg,:],axis=0)
 s1 = np.cov(train_features[train_pos,:].T,ddof =0)
 m1 = np.mean(train_features[train_pos,:],axis=0)
 sw = s0 + s1
 w = np.dot(np.linalg.inv(sw+0.001*np.eye(np.size(sw,0))),(m1-m0).T)

 features = np.dot(train_features, w)
 features_pos = features[train_pos]
 features_neg = features[train_neg]
 wm1 = np.mean(features_pos)
 wm2 = np.mean(features_neg)
 baise = -(wm1 + wm2)/2
 features_test = np.dot( test_features , w )+ baise
 predict_targets = np.sign(features_test)
 full = {"features": features, "features_test": features_test, "w": w, "baise": baise, "predict_targets":predict_targets} #输出字典
 return full

