from flask import Flask, render_template, redirect, url_for, request
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, IntegerField, DecimalField
from wtforms.validators import DataRequired, Length
import pandas as pd 
from sklearn import linear_model
import numpy as numpy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix  
import os



app = Flask(__name__, template_folder='template')

SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY



class DataForm(FlaskForm):
      gender = StringField('Gender', validators=[DataRequired()])
      age = DecimalField('Age', validators=[DataRequired()])
      urea = DecimalField('Urea', validators=[DataRequired()])
      cr = DecimalField('Chromium', validators=[DataRequired()])
      hba1c = DecimalField('Hemoglobin A1C', validators=[DataRequired()])
      chol = DecimalField('Cholestrol', validators=[DataRequired()])
      tg = DecimalField('Thyroglobulin', validators=[DataRequired()])
      hdl = DecimalField('High Density Lipoprotein', validators=[DataRequired()])
      ldl = DecimalField('Low Density Lipoprotein', validators=[DataRequired()])
      vldl = DecimalField('Very Low Density Lipoprotein', validators=[DataRequired()])
      bmi = DecimalField('BMI', validators=[DataRequired()])

      submit = SubmitField('Check')


@app.route("/", methods=['GET', 'POST'])
@app.route("/home", methods=['GET', 'POST'])

def home():
     
   form = DataForm()
   
     
   if request.method == "POST" and form.validate_on_submit():

      print("Worked.")
      DATA = [int(form.gender.data), form.age.data, form.urea.data, form.cr.data, form.hba1c.data, form.chol.data, form.tg.data, form.hdl.data, form.ldl.data, form.vldl.data, form.bmi.data]

      dataset = pd.read_csv('Hackathon.csv')
      dataset['CLASS'].value_counts()
      X = dataset.iloc[:, :-1].values
      y = dataset.iloc[:, -1].values
 
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
     

      scaler = StandardScaler()
      scaler.fit(X_train)
      X_train = scaler.transform(X_train)
      X_test = scaler.transform(X_test)

      classifier = KNeighborsClassifier(n_neighbors=5)
      classifier.fit(X_train, y_train)
      y_pred = classifier.predict(X_test) 

      cr = classification_report(y_test, y_pred)
     
      cm = confusion_matrix(y_test, y_pred)

     


     
      result = classifier.predict(scaler.transform([DATA]))

      if result == 0:
        message = "Non-Diabetic"
      elif result == 1:
        message = "Pre-diabetic"
      else: 
        message = "Diabetic"
      
            


      return render_template('index.html', form=form, prediction = message)

   return render_template('index.html', form=form)


if __name__ == '__main__':
    app.run(debug=True)


