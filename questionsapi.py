
# 1. Library imports
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query
from Questions import Question
import numpy as np
import pickle
import pandas as pd
from uuid import UUID, uuid4
import uuid
import os
from datetime import datetime

from fastapi_sessions.backends.implementations import InMemoryBackend
from fastapi_sessions.session_verifier import SessionVerifier
from fastapi_sessions.frontends.implementations import SessionCookie, CookieParameters


# 2. Create the app object
app = FastAPI()
classifier_q1 = pickle.load(open('logreg_q1.pkl', 'rb'))
classifier_q2= pickle.load(open('logreg_q2.pkl', 'rb'))
cv_q1 = pickle.load(open('transform_q1.pkl','rb'))
cv_q2 = pickle.load(open('transform_q2.pkl','rb'))


#cookie_params = CookieParameters()
# Uses UUID
#cookie = SessionCookie(
    #cookie_name="cookie",
    #identifier="general_verifier",
    #auto_error=True,
    #secret_key="DONOTUSE",
    #cookie_params=cookie_params,
#)
#backend = InMemoryBackend[UUID, SessionData]()


def add_to_log(sessionid,questionid, text,result,probabilty):
    now = datetime.now()
    now.strftime("%Y%m%d")
    path = now.strftime("%Y%m%d")+'_logs.csv'
    isExist = os.path.exists(path)
    if isExist:
        data =[[sessionid,questionid, text,result,probabilty, now]]
        df = pd.DataFrame(data, columns=['sessionid','questionid','text','classifiation_result','confidence','add_time'])
        df_old=pd.read_csv(path)
        frames = [df, df_old]
        result = pd.concat(frames)
        result=result[['sessionid','questionid','text','classifiation_result','confidence','add_time']]
        result.to_csv(path)
    else:
        data = [[sessionid,questionid, text,result,probabilty, now]]
        df = pd.DataFrame(data, columns=['sessionid','questionid','text','classifiation_result','confidence','add_time'])
        df.to_csv(path)

def get_session():
    with Session(engine) as session:
        yield session

# 3. Index route, opens automatically on http://127.0.0.1:8000



@app.get('/')
def index():
    return {'message': 'Hello, World'}




# 3. Expose the prediction functionality, make a prediction from the passed

@app.post('/predict_q')
def q1(data:Question):
    sessionid=uuid.uuid4()
    data = data.dict()
    text=data['text']
    questionid=data['questionid']
    print("Getting Session")
    session_id = Depends(get_session)
    #session_id=Depends(cookie)
    print("Session", session_id)
    if questionid==1:
        numpyArray = np.array([text])
        cv_q1 = pickle.load(open('transform_q1.pkl','rb'))
        x1=cv_q1.transform(numpyArray)
        classifier_q1 = pickle.load(open('logreg_q1.pkl', 'rb'))
        prediction_logreg = classifier_q1.predict(x1)
        prob=classifier_q1.predict_proba(x1)[:,1]
        add_to_log(sessionid,questionid, text,prediction_logreg,prob)
        if prediction_logreg[0]==0:
            return "0"
        if prediction_logreg[0]==1:
            return "1"
        if prediction_logreg[0]==2:
            return "2"
        if prediction_logreg[0]==3:
            return "3"
    if questionid==2:
        numpyArray = np.array([text])
        trf_data = cv_q2.transform(numpyArray)
        classifier_q2 = pickle.load(open('logreg_q2.pkl', 'rb'))
        prediction_logreg = classifier_q2.predict(trf_data)
        prob=classifier_q2.predict_proba(trf_data)[:,1]
        add_to_log(sessionid,questionid, text,prediction_logreg,prob)
        return prediction_logreg[0]





# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

#uvicorn questionsapi:app --reload
