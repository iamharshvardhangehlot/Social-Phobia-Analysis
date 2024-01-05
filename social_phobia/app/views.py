from django.http import JsonResponse
from django.shortcuts import render
import pickle
from sklearn.ensemble import RandomForestClassifier
from statistics import median

def index(request):
    return render(request, 'index.html')

def getPredictions(q1, q2, q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,q13,q14,q15,q16,q17):
    model=pickle.load(open('social_phobia_rf.pkl','rb'))

    a1 = median([q1, q2, q3,q4,q5,q6])
    a2 = median([q7,q8,q9,q10,q11,q12,q13])
    a3 = median([q14,q15,q16,q17])
    # prediction = model.predict(model.transform([q1,q2,q3]))

    # if prediction == 0:
    #     return "0"
    # elif prediction ==1:
    #     return "1"
    # elif prediction ==2:
    #     return "2"
    # elif prediction ==3:
    #     return "3"
    # elif prediction == 4:
    #     return "4"
    # elif prediction == 5:
    #     return "5"
    # else:
    #     return "error"

    if isinstance(model, RandomForestClassifier):
        # Reshape the input features into a 2D array
        input_features = [[a1, a2, a3]]

        # Make predictions using the model
        prediction = model.predict(input_features)

        # Convert the prediction to a string and return
        return str(prediction[0])

    else:
        return "Error: Model is not a RandomForestClassifier"

    

def result (request):
    # pclass = int(request.GET['pclass'])
    if request.method== 'GET' and request.headers.get('x-requested-with')=='XMLHttpRequest':
        q1 = int(request.GET['q1'])
        q2 = int(request.GET['q2'])
        q3 = int(request.GET['q3'])
        q4 = int(request.GET['q4'])
        q5 = int(request.GET['q5'])
        q6 = int(request.GET['q6'])
        q7 = int(request.GET['q7'])  
        q8 = int(request.GET['q8'])
        q9 = int(request.GET['q9'])
        q10 = int(request.GET['q10'])
        q11 = int(request.GET['q11'])
        q12 = int(request.GET['q12'])
        q13 = int(request.GET['q13'])
        q14 = int(request.GET['q14'])
        q15 = int(request.GET['q15'])
        q16 = int(request.GET['q16'])
        q17 = int(request.GET['q17'])
        result = getPredictions(q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,q13,q14,q15,q16,q17)

        percent = (int(result) + 1) * 100 // 5
        print("RESULT",percent, result)
        return JsonResponse({'result':result, 'percent':percent}) 
    else:
        return JsonResponse({'error':'Invalid Request'})