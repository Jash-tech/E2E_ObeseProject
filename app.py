from flask import Flask,request,render_template,jsonify
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline


application=Flask(__name__)

app=application



@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')
    
    else:
        data = CustomData(
            Gender=request.form.get('gender'),
            Age=float(request.form.get('age')),
            Height=float(request.form.get('height')),
            Weight=float(request.form.get('weight')),
            family_history_with_overweight=request.form.get('family_history'),
            FAVC=request.form.get('favc'),
            FCVC=float(request.form.get('fcvc')),
            NCP=float(request.form.get('ncp')),
            CAEC=request.form.get('caec'),
            SMOKE=request.form.get('smoke'),
            CH2O=float(request.form.get('ch2o')),
            SCC=request.form.get('scc'),
            FAF=float(request.form.get('faf')),
            TUE=float(request.form.get('tue')),
            CALC=request.form.get('calc'),
            MTRANS=request.form.get('mtrans')
        )
        
        final_new_data = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(final_new_data)

        results = round(pred[0], 2)

        return render_template('results.html', final_result=results)






if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)