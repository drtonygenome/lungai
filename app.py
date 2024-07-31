from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df=pd.read_csv("surveylungcancer.csv")
df =df.replace("M", '0')
df =df.replace("F", '1')
df =df.replace("NO", '0')
df =df.replace("YES", '1')

df['GENDER']= pd.to_numeric(df.GENDER)
df['group']= pd.to_numeric(df.group)
data=df


train_data = data.iloc[:-50]
eval_data = data.iloc[-50:]


X_train = train_data.drop(columns=["group"])
y_train = train_data["group"]

X_eval = eval_data.drop(columns=["group"])
y_eval = eval_data["group"]

# Chuẩn hóa dữ liệu sử dụng StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_eval_scaled = scaler.transform(X_eval)

loaded_model = tf.keras.models.load_model("lung_model.keras")

eval_loss, eval_acc = loaded_model.evaluate(X_eval_scaled, y_eval)
print(
    f"\nPrecision: {eval_acc * 100:.2f}%")

# ---------------APP-------------------- #
app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Get user input from the web form
        GENDER = float(request.form["GENDER"])
        AGE = int(request.form["AGE"])
        SMOKING = float(request.form["SMOKING"])
        YELLOW_FINGERS = int(request.form["YELLOW_FINGERS"])
        ANXIETY = float(request.form["ANXIETY"])
        PEER_PRESSURE = int(request.form["PEER_PRESSURE"])
        CHRONIC_DISEASE = float(request.form["CHRONIC_DISEASE"])
        FATIGUE  = int(request.form["FATIGUE"])
        ALLERGY  = float(request.form["ALLERGY"])
        WHEEZING = int(request.form["WHEEZING"])
        ALCOHOL_CONSUMING = float(request.form["ALCOHOL_CONSUMING"])
        COUGHING = int(request.form["COUGHING"])
        SHORTNESS_BREATH = float(request.form["SHORTNESS_BREATH"])
        SWALLOWING_DIFFICULTY = int(request.form["SWALLOWING_DIFFICULTY"])
        CHEST_PAIN = float(request.form["CHEST_PAIN"])
        

        # Create a DataFrame from user input
        input_data = pd.DataFrame({
            "GENDER": [GENDER],
            "AGE": [AGE],
            "SMOKING": [SMOKING],
            "YELLOW_FINGERS": [YELLOW_FINGERS],
            "ANXIETY": [ANXIETY],
            "PEER_PRESSURE": [PEER_PRESSURE],
            "CHRONIC_DISEASE": [CHRONIC_DISEASE],
            "FATIGUE": [FATIGUE],
            "ALLERGY": [ALLERGY],
            "WHEEZING": [WHEEZING],
            "ALCOHOL_CONSUMING": [ALCOHOL_CONSUMING],
            "COUGHING": [COUGHING],
            "SHORTNESS_BREATH": [SHORTNESS_BREATH],
            "SWALLOWING_DIFFICULTY": [SWALLOWING_DIFFICULTY]
            "CHEST_PAIN": [CHEST_PAIN]
            })

        # Scale the input data using the pre-trained scaler
        input_data_scaled = scaler.transform(input_data)

        # Make a prediction using the loaded model
        prediction = loaded_model.predict(input_data_scaled)
        predicted_classe = (prediction > 0.5).astype(int)
        print(input_data)
        print(prediction)
        print(predicted_classe)
        # Chuyển đổi giá trị prediction và predicted_classe thành chuỗi
        predicted_classe_str = str(predicted_classe[0][0])
        prediction_str = "{:.2f}%".format(prediction[0][0]*100)

        # Determine the result (group)
        result = "Possible" if predicted_classe[0][0] == 1 else "Impossible"

        return render_template("result.html", result=result, input_data=input_data, prediction=prediction_str, predicted_classe=predicted_classe_str)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=80)
