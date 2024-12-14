from utils.data.load_data import Fetcher
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
# from utils.analysis.evaluation import reidentify
import numpy as np
# synthcity absolute
# from synthcity.benchmark import Benchmarks
# from synthcity.plugins.core.constraints import Constraints
# from synthcity.plugins.core.dataloader import GenericDataLoader
# from synthcity.metrics import privacy, utility



adult = Fetcher('hospital') #adult
epsilon = 5
synth_adult = Fetcher('synth_hospital',epsilon=epsilon)

X= adult.dataset['X'].values
Y = adult.dataset['y'].values
    #Create pipeline
X_synth_adult = synth_adult.dataset['X'].values
y_synth_adult = synth_adult.dataset['y'].values

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,random_state=2)
model = MLPClassifier ()
model.fit(X_train, y_train)

y_pred_real = model.predict(X_test)
# y_pred_synth = model.predict(X_synth_adult)

print(f"Accuracy on real data: {accuracy_score(y_test, y_pred_real)}")
# print(f"Accuracy on synthetic data: {accuracy_score(y_synth_adult, y_pred_synth)}")


s_X_train, s_X_test, s_y_train, s_y_test = train_test_split(X_synth_adult, y_synth_adult, test_size=0.3,random_state=2)
model2 = MLPClassifier() #RandomForestClassifier()
model2.fit(s_X_train, s_y_train)

# y_synth_pred = model.predict(s_X_test)
y_s_pred_real = model2.predict(X_test)

# print(f"Synth Accuracy on synth data: {accuracy_score(s_y_test, y_synth_pred)}")
print(f"Synth Accuracy on real data: {accuracy_score(y_test, y_s_pred_real)}  for epsilon: {epsilon}")



# disclosure_count = 0
# for item in X:
#     for i in range(len(X_synth_adult)):
#         a = np.where(item == X_synth_adult[i])[0]
#         if len(a) == len(item) :
#             disclosure_count += 1

# disclosure_risk = disclosure_count / len(adult) # (adult.isin(synth_adult)).sum().sum() / adult.size
# print(f"Disclosure Risk: {disclosure_risk}")

# MI_count = 0
# for item in X_synth_adult:
#         a = np.where(item == X[i])[0]
#         if len(a) == len(item) :
#             MI_count += 1

# membership_inference_risk = MI_count / len(synth_adult) 
# print(f"Membership Inference Risk: {membership_inference_risk}")

# adult.describe()
# synth_adult.describe()

# membership_inference_risk = sum(adult.apply(tuple,1).isin(synth_adult.apply(tuple,1))) / len(adult)
# print(f"Membership Inference Risk: {membership_inference_risk}")

# utility_metrics = utility.evaluate(
#     real_data=adult,
#     synthetic_data=synth_adult
# )

# privacy_metrics = privacy.evaluate(
#     real_data=adult,
#     synthetic_data=synth_adult
# )

# # Print the results
# print("Utility Metrics:")
# print(utility_metrics)

# print("\nPrivacy Metrics:")
# print(privacy_metrics)
