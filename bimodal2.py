from pylab import *
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
import weibull
from scipy import stats
import math
from sklearn.metrics import mean_squared_error
import seaborn as sns
from windrose import WindroseAxes
def gauss(x,mu,sigma,A):
    return A*exp(-(x-mu)**2/2/sigma**2)

def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)
def weib(x,n,a):
    return (a / n) * (x / n)**(a - 1) * np.exp(-(x / n)**a)
# Inicializando variables
r = np.zeros(12)
r2 = np.zeros(12)
std_WV = np.zeros(12)
mean_WV = np.zeros(12)
rms = np.zeros(12)
rms2 = np.zeros(12)
i=1
ii= 0
j=0
Ener = np.zeros(12)
Energy = []
aux = np.zeros(154654)
Archivo = 'Qollpana150914-270818.csv'
df = pd.read_csv(Archivo, index_col= False)
print(max(df.index))
df["Fecha"] = pd.to_datetime(df["Fecha"])
for i in range(330,360,30):
    mascara = ((df['Viento - Direccion (°)']>i) & (df['Viento - Direccion (°)']<(i+31))) 
    dfaux = df[mascara]
    aux = np.zeros(dfaux.shape[0])
    auxt = np.zeros(df.shape[0])
    print(dfaux['Viento - Direccion (°)'])
    plt.subplots_adjust(hspace=0.9)
    plt.subplot(2,1,1)
    count, bins, ignored = plt.hist(dfaux["Viento - Velocidad (m/s)"],bins=range(0,23) )  
    # plt.close()
    data=dfaux["Viento - Velocidad (m/s)"]
    y,x,_=hist(data,range(0,23),label='data')
    print(count," ",bins)
    print(y, " " ,x)
    x=(x[1:]+x[:-1])/2 # for len(x)==len(y)
    analysis = weibull.Analysis(dfaux["Viento - Velocidad (m/s)"], unit = "m/s")
    analysis.fit(method='mle')
    # Capturando los parametros de weibull
    forma = analysis.stats[3]
    escala = analysis.stats[6]
    xx = np.linspace(min(dfaux["Viento - Velocidad (m/s)"]),max(dfaux["Viento - Velocidad (m/s)"]),sum(count))
    scale = count.max()/weib(xx,escala ,forma).max()
    plt.plot(xx, weib(xx,escala,forma)*scale,'-b', label = 'Weibull')
    # count, bins, ignored = plt.hist(dfaux["Viento - Velocidad (m/s)"],bins=range(0,int(dfaux["Viento - Velocidad (m/s)"].max()+2)))
    # Parametros de Bimodal
    params,cov=curve_fit(bimodal,x,y)
    sigma=sqrt(diag(cov))

    plot(xx,bimodal(xx,*params),color='green',lw=3,label='Bimodal')
    plt.xlabel("Vel. Viento [m/s]")
    plt.ylabel("Distribution")
    plt.title("Probability Distribution Function")
    legend()
    # plt.savefig("Distribucion Bimodal")

    #******************************************************************
    #			Generacion de Tablas de Frecuencia , PDF, y CDF
    #******************************************************************
    print(" ")
    print("Probabilidades")
    print(" ")
    probObs = count/sum(count)
    #print(probObs)
    probBim = bimodal(bins[1:],*params)/sum(bimodal(bins[1:],*params))
    #print(probBim)
    probWeib = weib(bins[1:],escala,forma)*scale/sum(weib(bins[1:],escala,forma)*scale)
    # print(probWeib)
    probAcBim = np.cumsum(probBim)
    probAcWeib = np.cumsum(probWeib)
    #print(probAcWeib)
    probAcReal = np.cumsum(probObs)
    #print(probAcReal)
    #******************************************************************
    #				Coeficiente de Correlaicon Pearson
    #******************************************************************
    r[ii-1], p = stats.pearsonr(probAcReal,probAcWeib)
    r2[ii-1], p2 = stats.pearsonr(probAcReal,probAcBim)
    print("correlacion r Bimodal= ", r2[ii-1])
    print("correlacion r weibull= ", r[ii-1])
    #******************************************************************
    #				RMSE Error
    #******************************************************************

    rms[ii-1] = sqrt(mean_squared_error(probAcReal, probAcWeib))
    rms2[ii-1] = sqrt(mean_squared_error(probAcBim,probAcReal))
    print("Weibull Error RMSE = ",rms[ii-1])
    print("Bimodal Error RMSE = ",rms2[ii-1])
    #******************************************************************
    #		Calculando la media y su desviacion estandar
    #******************************************************************
    # promedio
    std_WV[ii-1] = dfaux.loc[:,"Viento - Velocidad (m/s)"].std()
    print("std ",std_WV[ii-1])
    mean_WV[ii-1] = dfaux.loc[:,"Viento - Velocidad (m/s)"].mean()
    print("mean",mean_WV[ii-1])
    #******************************************************************
    #				Generando Tablas y graficos
    #******************************************************************
    Intervalo = ["0-1","1-2", "2-3", "3-4", "4-5", "5-6","6-7", "7-8", "8-9", "9-10", 
                        "10-11","11-12", "12-13", "13-14", "14-15", "15-16","16-17", "17-18",
                        "18-19", "19-20", "20-21",'21-22']
    StatData = {    'Intervalo': Intervalo,
                    'Real' : probObs,
                    'Weibull':probWeib,
                    'Bimodal': probBim,
                    'RealAcum': probAcReal,
                    'WeibullAcum': probAcWeib,
                    'BimodalAcum' : probAcBim 
                    }
    dfDist = pd.DataFrame(StatData, columns = ['Intervalo', 'Real', 'Weibull', 'Bimodal', 'RealAcum', 'WeibullAcum', 'BimodalAcum'])
    print(dfDist)
    plt.subplot(223)
    sns.regplot(x="RealAcum", y="BimodalAcum", data=dfDist)
    plt.text(0.2,0.9,r"$r^2 =$"+"{0:.4f}".format(r2[ii-1]),fontsize = 7)
    plt.text(0.2,0.7,r"$RMSE =$"+"{0:.4f}".format(rms2[ii-1]),fontsize = 7)
    plt.subplot(224)
    sns.regplot(x="RealAcum", y="WeibullAcum", data=dfDist)
    plt.text(0.2,0.9,r"$r^2 =$"+"{0:.4f}".format(r[ii-1]),fontsize = 7)
    plt.text(0.2,0.7,r"$RMSE =$"+"{0:.4f}".format(rms[ii-1]),fontsize = 7)
    plt.savefig("Qollpana"+str(i)+"_"+str(i+30)+".png",dpi=300)
    dfDist.to_csv("Qollpana"+str(i)+"_"+str(i+30)+".csv")
    plt.show() 
    plt.close()
    print(i)
    # dfstat = pd.DataFrame(StatData,columns= ['Fecha','V mean', 'V std','c', 'k','RWeibull', 'RRayleigh'])
    # print(dfstat.head())

    # with pd.ExcelWriter("StatMonths/StatResumen.xlsx") as writer:
    #     dfDist.to_excel(writer, sheet_name='Sheet1')
    count, bins, ignored = plt.hist(df['Viento - Direccion (°)'],bins=range(0,390,30) ) 
    #   Calculando el porcentaje de Energia Acumulada
    Ener[j]= (count[j])/sum(count)
    print(Ener[j])
    print(sum(count))
    plt.xlabel("Direccion del Viento (°)")
    plt.ylabel("Distribution")
    plt.title("Distribucion Direccion de Viento") 
    plt.savefig('Distribucion12Direccion.png', dpi =300)
    plt.show()
    plt.close()
    auxt = df["Viento - Velocidad (m/s)"]*df["Viento - Velocidad (m/s)"]*df["Viento - Velocidad (m/s)"]
    aux2t = sum(auxt) / df.shape[0]
    print(sum(auxt))
    print(df.shape)
    print(dfaux.shape)

    aux = dfaux["Viento - Velocidad (m/s)"]*dfaux["Viento - Velocidad (m/s)"]*dfaux["Viento - Velocidad (m/s)"]
    aux2 = sum(aux) / dfaux.shape[0]
    print(sum(aux))
    print(dfaux.shape[0])
#     Energ = (1 / sum(count))*(dfaux.loc[:,'w_33'])
#     Energy.append(Energ)
    # ax = WindroseAxes.from_ax()
    # ax.box(df['Viento - Direccion (°)'],df["Viento - Velocidad (m/s)"],normed = True, bins=np.arange(0, 23, 1) , nsector=12)
    # ax.set_legend()
    #plt.savefig("WRQollpana"+str(i)+"_"+str(i+30)+".png",dpi =300)
    # plt.show()
    # plt.close()
    j=j+1

print(Ener)
print(aux2)
print(aux2t)
print(dfaux["Viento - Velocidad (m/s)"])
Intervalo = ["0-30","30-60", "60-90", "90-120", "120-150", "150-180","180-210", "210-240", "240-270", "270-300", 
                        "300-330","330-360"]
StatData = {    'Sector': Intervalo,
                    'Porcentaje de Datos Agrupados' : Ener
            }
dfDist = pd.DataFrame(StatData, columns = ['Sector', 'Porcentaje de Datos Agrupados'])
# with pd.ExcelWriter("Regiones.xlsx") as writer:
#         dfDist.to_excel(writer, sheet_name='Sheet1')