
Folder Information:

- DataSet.csv : Dataset used for model building
- code.ipynb : Jupyter Notebook with the code for data preprocessing, visualization and model building
- recommend.ipynb : Jupyter Notebook with the code for recommendation system
- app.py : Flask API
- requirements.txt : Required libraries
- scripts.py : Data Preprocessing and Model Building
- README.md : Information about the API
- rf.pkl : Random Forest Model file
- scaler.pkl : Scaler file
- ohe.pkl : OneHotEncoder file

# API testing with Postman 

- step1 : Run app.py and Open Postman
- step2 : Select the request type 'POST'
- step3 : Enter the URL `http://127.0.0.1:5000/predict`
- step4 : In the Headers tab, enter the key as `Content-Type` and value as `application/json`
- step5 : Now Select the Body tab
- step6 : Select the raw radio button 
- step7 : Select the JSON (application/json) from the dropdown
- step8 : Enter the input data in JSON format

The following can features values can take only the following values:

`gender = Male/Female`
`payment_method = Cash/Credit Card/Bank Transfer/PayPal`

Ex:
{
    "gender":"Male",
    "tenure":5,
    "monthly_charges":100,
    "total_charges": 100,
    "service_usage1":21,
    "service_usage2":30,
    "service_usage3":40,
    "payment_method":"Cash"
}

- step9 : Click on the Send button
- step10 : The output will be displayed in the Body section of the response

# API testing with cURL

- step1 : Run app.py
- step2 : Open the terminal
- step3 : Enter the following command in the powershell
```
curl -Method POST -Uri http://127.0.0.1:5000/predict `
    -ContentType "application/json" `
    -Body '{"gender": "Male", "tenure": 5, "monthly_charges": 100, "total_charges": 100, "service_usage1": 21, "service_usage2": 30, "service_usage3": 40,  "payment_method": "Cash"}'

```
- step4 : The output will be displayed in the terminal