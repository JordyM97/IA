from tkinter import *
from tkinter import messagebox
from sklearn.linear_model import LogisticRegression
from SentAnaLogisticRegression import *
class Aplicacion():
    resultado=""
    def __init__(self):
        self.classifier=Classifier()
        print(self.classifier.score)
        self.root = Tk()
        self.root.resizable(width=False, height=False)
        self.root.geometry("1000x700")
        self.root.title('Analisis de sentimientos de Tweets')
        self.lbl1 = Label(self.root, text="Modelo de Regresión Logaritmica para el análisis sentimental de tweets",font=("Helvetica", 30),fg="red")
        self.lbl1.pack(pady=10)
        self.lbl2 = Label(self.root, text="Efectividad del modelo: "+str(self.classifier.score),font=("Helvetica", 20),fg="black")
        self.lbl2.pack(pady=10)
        self.lbl3 = Label(self.root, text="Ingresa un tweet relacionados al coronavirus",font=("Helvetica", 15),fg="black")
        self.lbl3.pack(pady=5)
        self.TextArea = Entry(self.root, width=50,bg="sky blue")
        self.TextArea.pack()
        self.bt= Button(self.root, text="Analizar tweet",bg="black",command=self.imprimir)
        self.bt.pack(side=TOP, padx=10, pady=5)
        self.lbl4 = Label(self.root, text="Resultado: "+self.resultado,font=("Helvetica", 20),fg="black")
        self.lbl4.pack(pady=10)


        self.root.mainloop()
    
    def imprimir(self):
        test = self.classifier.vectorizer.transform([self.TextArea.get()])
        result= self.classifier.classifier.predict(test);
        print(result[0])

        
        
Aplicacion()