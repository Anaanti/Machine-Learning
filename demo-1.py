import numpy as np
import pickle
loaded_model= pickle.load(open(r"C:/Users/yatia/Desktop/Machine Learning/Loan Prediction/loan_status_prediction_model.sav",'rb'))

input_data = (1,0,0,1,0,2014,1929,74,360,1,2)
input_data_as_numpy = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy.reshape(1,-1)
prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print("Loan is not Approved.")
else:
  print("Loan is Approved.")