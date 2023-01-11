
# 1. Library imports
import uvicorn
from fastapi import FastAPI
from Questions import Question1, Question2
import numpy as np
import pickle
import pandas as pd
# 2. Create the app object
app = FastAPI()
classifier_q1 = pickle.load(open('logreg_q1.pkl', 'rb'))
classifier_q2= pickle.load(open('logreg_q2.pkl', 'rb'))
cv_q1 = pickle.load(open('transform_q1.pkl','rb'))
cv_q2 = pickle.load(open('transform_q2.pkl','rb'))
# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}



# 3. Expose the prediction functionality, make a prediction from the passed

@app.post('/predict_q1')
def q1(data:Question1):
    data = data.dict()
    text=data['text']
    numpyArray = np.array([text])
    cv_q1 = pickle.load(open('transform_q1.pkl','rb'))
    x1=cv_q1.transform(numpyArray)
    classifier_q1 = pickle.load(open('logreg_q1.pkl', 'rb'))
    prediction_logreg = classifier_q1.predict(x1)
    if prediction_logreg[0]==0:
        return "0"
    if prediction_logreg[0]==1:
        return "1"
    if prediction_logreg[0]==2:
        return "2"
    if prediction_logreg[0]==3:
        return "3"




@app.post('/predict_q2')
def q2(data:Question2):
    data = data.dict()
    text=data['text']
    numpyArray = np.array([text])
    trf_data = cv_q2.transform(numpyArray)
    classifier_q2 = pickle.load(open('logreg_q2.pkl', 'rb'))
    prediction_logreg = classifier_q2.predict(trf_data)
    return prediction_logreg[0]


# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

#uvicorn questionsapi:app --reload
