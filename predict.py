import joblib
import numpy as np
features = np.loadtxt("X.txt")
samples = features.shape[0]
pre = []
for i in range(10):
    i = i+1
    Model = joblib.load("Model-"+str(i)+".pkl")
    pre.append(Model.predict(features))
pre = np.array(pre)
pre = pre.reshape(samples,10)
print("The final result is: ",np.array(pre).mean(axis=1))

