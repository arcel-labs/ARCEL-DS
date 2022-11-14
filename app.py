import pandas as pd
from flask import Flask
from flask import request
import pickle
import json

app = Flask(__name__)

@app.route('/model', methods=['POST'])

def request_args():
    feature_dict = request.get_json()
    print(feature_dict)
    print(type(feature_dict))
    #feature_dict = pd.json_normalize(feature_dict)
    #feature_dict = pd.read_json(feature_dict)
    
    response = get_model_response(feature_dict)
  

    return response

    #age = request.args.get('age', '')
    #mother_education =  request.args.get('age', '')
    #father_education = request.args.get('father_education', '')
    #travel_time = request.args.get('travel_time', '')
    #study_time = request.args.get('study_time', '')
    #Failures = request.args.get('Failures', '')
    #family_relationship = request.args.get('family_relationship', '')
    #free_time_after_school = request.args.get('free_time', '')
    #number_of_school_absences = request.args.get('absences', '')
    #total_grade = request.args.get('grade_mean', '')

def predict(X, model):
    prediction = model.predict(X)[0]
    return prediction

def get_model_response(feature_dict):
    filename = 'boost_model.sav'
    model = pickle.load(open(filename, 'rb'))

    X = pd.DataFrame(feature_dict, index=[0])
    prediction = predict(X, model)
    if prediction < 6:
        label = "Provavel Reprovação"
    else: 
        label = "Provável Aprovação"
    return {
        'label': label,
        'prediction': int(prediction)
    }

    request_args()

if __name__ == '__main__':
    app.run(host='0.0.0.0')
    
