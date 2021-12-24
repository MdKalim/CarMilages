from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd

import joblib
reloadModel = joblib.load('./model/RFModelforMPG.pkl')
# Create your views here.
def index(request):
    context ={'a1':'Hello World'}
    return render(request,'index.html',context)
    # return HttpResponse({'a1':1})

def predictMPG(request):
    print(request)
    if request.method == 'POST':
        temp={}
        temp['cylinders']=request.POST.get('cylinderVal')
        temp['displacement']=request.POST.get('dispVal')
        temp['horsepower']=request.POST.get('hrsPwrVal')
        temp['weight']=request.POST.get('weightVal')
        temp['acceleration']=request.POST.get('accVal')
        temp['model year']=request.POST.get('modelVal')
        temp['origin']=request.POST.get('originVal')
        
    testDtaa=pd.DataFrame({'x':temp}).transpose()
    score = reloadModel.predict(testDtaa)[0]
    print(score)
    context ={'score':score}
    return render(request,'index.html',context)