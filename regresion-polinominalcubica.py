import numpy as np
import math as mt

class Dataset():
    def __init__(self,x,y) -> None:
        self.x = x
        self.y = y  

class Matematicas(Dataset):
    def __init__(self, x, y) -> None:
        super().__init__(x, y)

    # aqui hice la matriz que tiene 
    def MatrizX(self):
        #array x
        x = np.array(self.x)
        #x0
        ns1 = np.ones((len(x), 1))
        #x2
        x_squand = np.square(self.x)
        #x3
        x_cubic = np.power(self.x,3)
        join = np.concatenate((ns1, x.reshape(-1, 1), x_squand.reshape(-1, 1), x_cubic.reshape(-1, 1)), axis=1)
        #cambiar el formato par aver mejor
        np.set_printoptions(formatter={'float': '{:.6f}'.format})
        return join
    
    # Saque la traspuesta

    def traspuesta(self):
        matrixTras = self.MatrizX()
        traspuestaM = np.transpose(matrixTras)
        return traspuestaM
    
    #multiplicame la matrix por la traspuesta

    def MultiMatrixTras(self):
        varMatrixNormal = self.MatrizX()
        varMatrixTras = self.traspuesta()
        Total = np.dot(varMatrixTras, varMatrixNormal)
        return Total
    
    #Inversa de la matrix
    def InverMatrix(self):
        varMultiMat = self.MultiMatrixTras()
        inver = np.linalg.inv(varMultiMat)
        return inver

    #multipllicar traspuesta por la inversa

    def MultiXTrasPInver(self):
        varInverMatrix = self.InverMatrix()
        varTrasMatrix = self.traspuesta()
        total = np.dot(varInverMatrix, varTrasMatrix)
        return total

    # la ultima parte de la formula multiplicar por y

    def MultiPY(self):
        vary = np.array(self.y)
        varMultiTrasPinver = self.MultiXTrasPInver()
        total = np.dot(varMultiTrasPinver, vary)
        #total = np.inner() es lo mismo que lo de arriva 
        return total
    

class main(Matematicas):
    def __init__(self, x, y) -> None:
        super().__init__(x, y)


    def passTList(self):
        listaR2 =[] 
        listaR = self.MultiPY()
        for i  in listaR:
            listaR2.append(i)            
        return listaR2

    def Prediccion(self,numero):
        varlist = self.passTList()
        listX = [numero, numero ** 2,numero **3]
        calculo1 = varlist[1] * listX[0] 
        calculo2 = varlist[2] * listX[1]
        calculo3 = varlist[3] * listX[2]
        Total = varlist[0]  + calculo1 + calculo2 + calculo3
        return Total

    def R_cuadrado(self):
        y_hat = np.asarray([self.Prediccion(x_val) for x_val in self.x])
        y_mean = np.mean(self.y)
        SS_res = np.sum((self.y - y_hat) ** 2)
        SS_tot = np.sum((self.y - y_mean) ** 2)
        R2 = 1 - (SS_res / SS_tot)
        return R2



#x
x=[108,115,106,97,95,91,97,83,83,78,54,67,56,53,61,115,81,78,30,45,99,32,25,28,90,89]
#y
y=[95,96,95,97,93,94,95,93,92,86,73,80,65,69,77,96,87,89,60,63,95,61,55,56,94,93]



print("\n-----------Y\U0001F600---R\u00B2----------------")

objet_0 = main(x,y)
print(objet_0.passTList())
print(f"La R\u00B2 = {objet_0.R_cuadrado()}")

print("\n")

print("\n-----------------------1\U0001F600---------------------")
obj_1 = main(x,y)
print(obj_1.Prediccion(108))

print("\n-----------------------2\U0001F600-----------------------")
objet_2 = main(x,y)
print(objet_2.Prediccion(31))


print("\n-----------------------3\U0001F600-----------------------")
objet_3 = main(x,y)
print(objet_3.Prediccion(20))


print("\n-----------------------4\U0001F600-----------------------")
objet_4 = main(x,y)
print(objet_4.Prediccion(40))

print("\n-----------------------5\U0001F600------------------------")
objet_5 = main(x,y)
print(objet_5.Prediccion(39))

print("\n\n")






