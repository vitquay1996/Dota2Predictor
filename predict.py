from sklearn.externals import joblib
import numpy as np
import json
import requests

team = ["Radiant", "Dire"]
clf = joblib.load('predictor_0.4.pkl')

# # Sample only
# sample = json.loads('["30","37.1","30","37.4","100","54.2","48.3","66.2","70.7","67.7"]')
# sample_np = np.array([(float(a)-50)/50 for a in sample])
# print("Predicted winner: ", team[clf.predict(sample_np.reshape(1,-1))[0]])


# Real
match_id = "3703866531"
response = requests.post("https://hacknroll18.herokuapp.com/dota", json={"matchId":  match_id}).json()
print(response)
np_vector = np.array([(float(a)-50)/50 for a in response['players']])
result = response['winner']
# print(clf.feature_importances_)
print("Predicted winner: ", team[clf.predict(np_vector.reshape(1,-1))[0]])
print("Real winner: ", team[result])