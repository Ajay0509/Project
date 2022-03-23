from flask import Flask, render_template, flash, request
import torch


model=torch.load('D:\\Academics\\Major Project\\models\\model_scripted.pt')


# App configuration
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/diagnosis", methods=['POST'])
def diagnosis():
    name = request.form['name']
    age = request.form['age']
    pregnant = request.form['pregnant']
    insulin = request.form['insulin']
    bmi = request.form['bmi']
    pedigree = request.form['pedigree']
    glucose = request.form['glucose']
    bp = request.form['bp']
    skinthick=request.form['skinthick']
    
    L=[pregnant,glucose,bp,skinthick,insulin,bmi,pedigree,age]
    L=[float(i) for i in L]
    
    new_data=torch.tensor(L)
    with torch.no_grad():
    
        if(model(new_data).argmax().item()):
            return render_template("positive.html", result="true")
            
        else:
            return render_template("negative.html", result="true")
        
    
        
if __name__ == "__main__":
    app.run()