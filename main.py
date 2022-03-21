from flask import Flask, render_template, request, Response
import pickle
import pandas as pd
import xgboost
import numpy as np


model = pickle.load(open('HR_Retention_Prediction.sav', 'rb'))

app = Flask(__name__)


@app.route('/')
def man():
    return render_template('index.html')



@app.route('/', methods=['POST','GET'])
def prediction():
    """
    * method: prediction_route_client
    * description: method to call prediction route
    * return: none
    *
    *
    * Parameters
    *   None
    """
    try:


        print('Test')

        if request.method == 'POST':
            satisfaction_level = request.form.get('satisfaction_level')
            last_evaluation = request.form.get("last_evaluation")
            number_project = request.form.get("number_project")
            average_montly_hours = request.form.get("average_montly_hours")
            time_spend_company = request.form.get("time_spend_company")
            work_accident = request.form.get("work_accident")
            promotion_last_5years = request.form.get("promotion_last_5years")
            salary = request.form.get("salary")
            if(salary=='low'):
                n_salary=0
            elif(salary=='medium'):
                n_salary = 1
            else:
                n_salary = 2



            data = np.array([[satisfaction_level, last_evaluation, number_project, average_montly_hours,
                              time_spend_company, work_accident, promotion_last_5years, n_salary]], dtype='float64')
            print(data)
            output = model.predict(data)
            print('output : '+str(output))
            if(output==1):
                return Response("Prediction: Employee had left the organization")
            else:
                return Response("Prediction: Employee is still serving the organization")
    except ValueError:
        return Response("Error Occurred!v %s" % ValueError)
    except KeyError:
        return Response("Error Occurred!k %s" % KeyError)
    except Exception as e:
        return Response("Error Occurred!e %s" % e)


if __name__ == "__main__":
    app.run(debug=True)
