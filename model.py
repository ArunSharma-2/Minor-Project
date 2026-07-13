import importlib
from sklearn.model_selection import train_test_split

class Model:

    def __init__(self,modulename,packagename=""):
        self.module = modulename
        self.package = packagename
        try:
            m = importlib.import_module(self.module)
            if self.package != "": 
                att = getattr(m,self.package)
            
            print("success importing",self.module)
        except ImportError:
            print(self.module,"is not installed")
        

    def train(self,X_train,y_train):
        Model.train(X_train,y_train)
    def result():
        pass
    def graph():
        pass