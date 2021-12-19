from multiprocessing import Process,Pool
import copy
import time
import os
import threading
from pathlib import Path
from joblib import dump, load
import re
import pandas as pd
from sklearn.metrics import r2_score,classification_report,f1_score,confusion_matrix




def _fit_pipeline(pipe,xtrain,xtest,ytrain,ytest):
    model_folder = './saved_models/'
    file_name = re.sub('[^a-zA-Z0-9 \n\.]', '', list(pipe.named_steps.keys())[-1])
    print(file_name)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    if not Path(model_folder+file_name+'.joblib').is_file():
        start_time = time.time()
        logs = 'Model '+file_name+'\n'
        print('Training started')
        pipe.fit(xtrain, ytrain)
        predictions = pipe.predict(xtest)
        execution_time = (time.time() - start_time)
        #data = {'y_Actual': ytest,'y_Predicted': predictions}
        #df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
        #confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
        logs = logs + f'train duration {execution_time} \n'
        r2score = r2_score(ytest, predictions)
        logs = logs + f'r2 score: {r2score}\n'
        try:
            _confusion_matrix = confusion_matrix(ytest, predictions,normalize='true')
            logs = logs + f'Confusion matrix \n {_confusion_matrix} \n'
            f1score = f1_score(ytest, predictions,average='weighted')
            logs = logs + f'f1 score:  {f1score}\n '
        except:
            pass
        print(logs)
        with open('./logs'+file_name+'.txt', 'w') as f:
            f.write(logs)
        dump(pipe, model_folder+file_name+'.joblib')
    print('Model trained')