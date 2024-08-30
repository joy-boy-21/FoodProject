import numpy as np
from sklearn.linear_model import LinearRegression
from flask import Flask, request, jsonify, render_template

data = {
    'age': [5, 10, 15, 20, 30, 40, 50, 60, 70],
    'protein': [20, 30, 45, 50, 55, 60, 65, 70, 70],
    'glucose': [100, 130, 150, 180, 200, 220, 230, 210, 190],
    'fats': [30, 40, 55, 70, 80, 85, 90, 85, 75],
    'carbohydrates': [130, 160, 180, 200, 220, 240, 260, 240, 220],
    'vitamin_a': [300, 400, 600, 700, 900, 900, 850, 800, 750],
    'vitamin_b6': [0.5, 0.6, 1.0, 1.3, 1.3, 1.7, 1.7, 1.7, 1.5],
    'vitamin_b12': [1.2, 1.5, 2.0, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4],
    'vitamin_c': [45, 50, 60, 75, 90, 90, 85, 80, 75],
    'vitamin_d': [10, 15, 15, 15, 15, 20, 20, 20, 20],
    'vitamin_e': [6, 7, 8, 9, 10, 15, 15, 15, 15],
    'vitamin_k': [30, 55, 60, 75, 90, 120, 120, 120, 120],
    'minerals': [500, 800, 1000, 1200, 1300, 1400, 1500, 1400, 1300]
}

X = np.array(data['age']).reshape(-1, 1)
y_protein = np.array(data['protein'])
y_glucose = np.array(data['glucose'])
y_fats = np.array(data['fats'])
y_carbohydrates = np.array(data['carbohydrates'])
y_vitamin_a = np.array(data['vitamin_a'])
y_vitamin_b6 = np.array(data['vitamin_b6'])
y_vitamin_b12 = np.array(data['vitamin_b12'])
y_vitamin_c = np.array(data['vitamin_c'])
y_vitamin_d = np.array(data['vitamin_d'])
y_vitamin_e = np.array(data['vitamin_e'])
y_vitamin_k = np.array(data['vitamin_k'])
y_minerals = np.array(data['minerals'])

protein_model = LinearRegression().fit(X, y_protein)
glucose_model = LinearRegression().fit(X, y_glucose)
fats_model = LinearRegression().fit(X, y_fats)
carbohydrates_model = LinearRegression().fit(X, y_carbohydrates)
vitamin_a_model = LinearRegression().fit(X, y_vitamin_a)
vitamin_b6_model = LinearRegression().fit(X, y_vitamin_b6)
vitamin_b12_model = LinearRegression().fit(X, y_vitamin_b12)
vitamin_c_model = LinearRegression().fit(X, y_vitamin_c)
vitamin_d_model = LinearRegression().fit(X, y_vitamin_d)
vitamin_e_model = LinearRegression().fit(X, y_vitamin_e)
vitamin_k_model = LinearRegression().fit(X, y_vitamin_k)
minerals_model = LinearRegression().fit(X, y_minerals)

app = Flask(__name__)

def calculate_suggestions(age):
    age_array = np.array([[age]])

    protein = protein_model.predict(age_array)[0]
    glucose = glucose_model.predict(age_array)[0]
    fats = fats_model.predict(age_array)[0]
    carbohydrates = carbohydrates_model.predict(age_array)[0]
    vitamin_a = vitamin_a_model.predict(age_array)[0]
    vitamin_b6 = vitamin_b6_model.predict(age_array)[0]
    vitamin_b12 = vitamin_b12_model.predict(age_array)[0]
    vitamin_c = vitamin_c_model.predict(age_array)[0]
    vitamin_d = vitamin_d_model.predict(age_array)[0]
    vitamin_e = vitamin_e_model.predict(age_array)[0]
    vitamin_k = vitamin_k_model.predict(age_array)[0]
    minerals = minerals_model.predict(age_array)[0]

    return {
        'protein_g_day': round(protein, 2),
        'glucose_g_day': round(glucose, 2),
        'fats_g_day': round(fats, 2),
        'carbohydrates_g_day': round(carbohydrates, 2),
        'vitamin_a_mg_day': round(vitamin_a, 2),
        'vitamin_b6_mg_day': round(vitamin_b6, 2),
        'vitamin_b12_mg_day': round(vitamin_b12, 2),
        'vitamin_c_mg_day': round(vitamin_c, 2),
        'vitamin_d_mg_day': round(vitamin_d, 2),
        'vitamin_e_mg_day': round(vitamin_e, 2),
        'vitamin_k_mg_day': round(vitamin_k, 2),
        'minerals_mg_day': round(minerals, 2)
    }


@app.route('/')
def index():
    return render_template('chat.html')  

@app.route('/get_suggestions', methods=['POST'])
def get_suggestions():
    try:
        user_input = request.json
        age = int(user_input.get('age'))  
        suggestions = calculate_suggestions(age)

        return jsonify({
            'age': age,
            **suggestions
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
   
