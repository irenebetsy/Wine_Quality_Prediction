from Libraries import Flask,render_template,request,pd,json,traceback,jsonify

# Import the Predictor class (make sure it's defined in predictor_module.py)
from Prediction import Wine_Quality


app = Flask(__name__)
predictor = Wine_Quality()

@app.route('/', methods=['GET'])
def index():
    # Render an index.html template
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    """# Get JSON data from the form
    json_data = request.get_json()
    # Pass the JSON data to the result.html template, which will render the table
    return render_template('result1.html', data=json_data)"""
    try:
        # Get JSON input from the request
        input_json = request.json
        input_data = pd.DataFrame(input_json)
        print(input_data)

        # Convert JSON input to a Pandas DataFrame
        #print(f'About to call predict, data received: {input_json}')

        # Call the API to get predictions
        #response = requests.post('http://localhost:5000/predict_api', json=input_json)
        result_str = predictor.callPredict(input_data)
        #result_json = response.json()
        print(result_str) 
        json_data = json.loads(result_str)
        print(type(json_data))
        #result_json = [{"area":"5000","bedrooms":"3","bathrooms":"1","stories":"1","mainroad":"yes","guestroom":"no","basement":"no","hotwaterheating":"no","airconditioning":"yes","parking":"2","prefarea":"yes","furnishingstatus":"furnished","Predicted":5466075.4811099423}]
        print("******************** About to call result template ********************")
        return render_template('result.html', data=json_data)
        #return result_json
    except Exception as e:
        print(e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(port=5001,debug=True)
