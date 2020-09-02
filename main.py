from tkinter import *
from sklearn.linear_model import LogisticRegression
from SentAnaLogisticRegression import *
class Aplicacion():
    def __init__(self):
        
        self.classifier=Classifier()
        print(self.classifier.score)
        print(self.classifier.X_train)
        self.root = Tk()
        self.resultado= StringVar(self.root,value="Resultado: ")
        self.root.resizable(width=True, height=False)
        self.root.geometry("750x700")
        self.root.config(bg="brown")
        self.root.title('Analisis de sentimientos de Modalidad Virtual')
        self.lbl1= Label(self.root,text="Modelo de Regresión Logística \n en Análisis Sentimental de tweets respecto a la modalidad virtual",font=("Helvetica",30),fg="white",bg="brown")
        self.lbl1.pack(pady=50)
        self.lbl2 = Label(self.root, text="Efectividad del modelo: "+str("{:.2f}".format(self.classifier.score)),font=("Helvetica", 20),fg="black")
        self.lbl2.pack(pady=30)
        self.lbl3 = Label(self.root, text="Ingresa un tweet relacionados al coronavirus",font=("Helvetica", 15),fg="black")
        self.lbl3.pack(pady=10)
        self.TextArea = Entry(self.root, width=50,bg="white",bd=5,highlightcolor="black")
        self.TextArea.pack()
        self.bt= Button(self.root, text="Analizar tweet",bg="light blue",command=self.imprimir)
        self.bt.pack(side=TOP, padx=10, pady=20)
        self.lbl4 = Label(self.root, textvariable=self.resultado,font=("Helvetica", 20),fg="black")
        self.lbl4.pack(pady=10)


        self.root.mainloop()
    
    def imprimir(self):
        resultado=""
        test = self.classifier.vectorizer.transform([self.TextArea.get()])
        result= self.classifier.classifier.predict(test);
        if(result[0]==-1):
            resultado="Negativo"
        elif (result[0]==0):
            resultado="Neutro"
        else:
            resultado="Positivo"
        self.resultado.set("Resultado: "+resultado)

        
        
Aplicacion()