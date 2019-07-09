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

#******************************************************************
#					Datos Observados
#******************************************************************

Ener = np.zeros(12)
Archivo = '/home/opti3040a/Documentos/WRF/Qollpana150914-270818.csv'
df = pd.read_csv(Archivo, index_col= False)
df["Fecha"] = pd.to_datetime(df["Fecha"])
df['Fecha'] = df['Fecha'] + pd.Timedelta(hours=4)   # Corrigiendo UTC-4
print(max(df.index))
mask = df.Fecha.dt.year == 2018
dfvar = df.loc[mask]
mask = dfvar.Fecha.dt.month == 3
dfaux = dfvar.loc[mask]
print(dfaux.head())
#******************************************************************
#		Filtrando los datos cada hora
#******************************************************************
# Filtrando Fecha cada hora
print(dfaux.Fecha.dt.hour.head(10))
# Generando la mascara de intervalos de 10 minutos
Int_10 = dfaux.Fecha.dt.minute == 0
print(dfaux.loc[Int_10].reset_index())
df2 = dfaux.loc[Int_10].reset_index()
print(df2.info())
print(df2['Fecha'])
dfaux = df2
# ******************************************************************
					
# ******************************************************************

aux = np.zeros(dfaux.shape[0])
auxt = np.zeros(df.shape[0])
plt.subplots_adjust(hspace=0.9)
plt.subplot(2,1,1)
count, bins, ignored = plt.hist(dfaux["Viento - Velocidad (m/s)"],bins=range(0,23),density=True)  

data=dfaux["Viento - Velocidad (m/s)"]
y,x,_=hist(data,range(0,23),label='Datos Observados',density=True)
print(count," ",bins)
print(y, " " ,x)
x=(x[1:]+x[:-1])/2 # for len(x)==len(y)
analysis = weibull.Analysis(dfaux["Viento - Velocidad (m/s)"], unit = "m/s")
analysis.fit(method='mle')
# Capturando los parametros de weibull
forma = analysis.stats[3]
escala = analysis.stats[6]
xx = np.linspace(min(dfaux["Viento - Velocidad (m/s)"]),max(dfaux["Viento - Velocidad (m/s)"]),sum(count))
print(x)

plt.plot(x, weib(x,escala,forma),'-b', label = 'Weibull')
#******************************************************************
#					Datos simulados
#******************************************************************
Archivo = 'df_marzoCor2.csv'
dfs = pd.read_csv(Archivo, index_col= False)
dfauxs = dfs
print(dfauxs.head())

counts, binss, ignoreds = plt.hist(dfauxs["W10s_3km"],bins=range(0,23),density=True,color='green',alpha=0.2)  

datas=dfauxs["W10s_3km"]
ys,xs,_=hist(datas,range(0,23),label='datos WRF',density=True,alpha=0.2)
print(counts," ",binss)
print(ys, " " ,xs)
xs=(xs[1:]+xs[:-1])/2 # for len(x)==len(y)
analysiss = weibull.Analysis(dfauxs["W10s_3km"], unit = "m/s")
analysiss.fit(method='mle')
# Capturando los parametros de weibull
formas = analysiss.stats[3]
escalas = analysiss.stats[6]
xxs = np.linspace(min(dfauxs["W10s_3km"]),max(dfauxs["W10s_3km"]),sum(counts))
print(xs)
scales = counts.max()/weib(xs,escalas ,formas).max()

plt.plot(xs, weib(xs,escalas,formas),'-r', label = 'WeibullSim')
legend()

#******************************************************************
#			Calculando RMSE para simulados vs Observados
#******************************************************************


# plt.savefig("Distribucion Bimodal")

#******************************************************************
#			Generacion de Tablas de Frecuencia , PDF, y CDF
#******************************************************************
print(" ")
print("Probabilidades")
print(" ")
probObs = count/sum(count)
probWeib = weib(bins[1:],escala,forma)/sum(weib(bins[1:],escala,forma))
probAcWeib = np.cumsum(probWeib)
probAcReal = np.cumsum(probObs)
# Simulada WRF
print(" ")
print("Probabilidades")
print(" ")
probObss = counts/sum(counts)
probWeibs = weib(binss[1:],escalas,formas)/sum(weib(binss[1:],escalas,formas))
probAcWeibs = np.cumsum(probWeibs)
probAcReals = np.cumsum(probObss)


#******************************************************************
#				Coeficiente de Correlaicon Pearson
#******************************************************************
r, p = stats.pearsonr(probAcReal,probAcWeib)
print("correlacion r weibull= ", r)
#******************************************************************
#				RMSE Error
#******************************************************************

rms = sqrt(mean_squared_error(probAcReal, probAcWeib))
print("Weibull Error RMSE = ",rms)

#******************************************************************
#		Calculando la media y su desviacion estandar
#******************************************************************
# promedio
std_WV = dfaux.loc[:,"Viento - Velocidad (m/s)"].std()
print("std ",std_WV)
mean_WV = dfaux.loc[:,"Viento - Velocidad (m/s)"].mean()
print("mean",mean_WV)

# simulados WRF

rs, ps = stats.pearsonr(probAcReals,probAcWeibs)
print("correlacion r weibull= ", rs)
#******************************************************************
#				RMSE Error
#******************************************************************

rmss = sqrt(mean_squared_error(probAcReals, probAcWeibs))
print("Weibull Error RMSE = ",rmss)

#******************************************************************
#		Calculando la media y su desviacion estandar
#******************************************************************
# promedio
std_WVs = dfaux.loc[:,"Viento - Velocidad (m/s)"].std()
print("std ",std_WVs)
mean_WVs = dfaux.loc[:,"Viento - Velocidad (m/s)"].mean()
print("mean",mean_WVs)

#******************************************************************
#				Generando Tablas y graficos
#******************************************************************
Intervalo = ["0-1","1-2", "2-3", "3-4", "4-5", "5-6","6-7", "7-8", "8-9", "9-10", 
                    "10-11","11-12", "12-13", "13-14", "14-15", "15-16","16-17", "17-18",
                    "18-19", "19-20", "20-21",'21-22']
StatData = {    'Intervalo': Intervalo,
                'Real' : probObs,
                'Weibull':probWeib,
                'RealAcum': probAcReal,
                'WeibullAcum': probAcWeib,
                }
dfDist = pd.DataFrame(StatData, columns = ['Intervalo', 'Real', 'Weibull', 'RealAcum', 'WeibullAcum'])
print(dfDist)
# simulado WRF
Intervalo = ["0-1","1-2", "2-3", "3-4", "4-5", "5-6","6-7", "7-8", "8-9", "9-10", 
                    "10-11","11-12", "12-13", "13-14", "14-15", "15-16","16-17", "17-18",
                    "18-19", "19-20", "20-21",'21-22']
StatData = {    'Intervalo': Intervalo,
                'Real' : probObss,
                'Weibull':probWeibs,
                'RealAcum': probAcReals,
                'WeibullAcum': probAcWeibs,
                }
dfDists = pd.DataFrame(StatData, columns = ['Intervalo', 'Real', 'Weibull', 'RealAcum', 'WeibullAcum'])



plt.subplot(223)
sns.regplot(x="RealAcum", y="WeibullAcum", data=dfDist)
plt.text(0.2,0.9,r"$r^2 =$"+"{0:.4f}".format(r),fontsize = 7)
plt.text(0.2,0.7,r"$RMSE =$"+"{0:.4f}".format(rms),fontsize = 7)
plt.title('Datos Observados')
plt.subplot(224)
sns.regplot(x="RealAcum", y="WeibullAcum", data=dfDists)
plt.text(0.2,0.9,r"$r^2 =$"+"{0:.4f}".format(rs),fontsize = 7)
plt.text(0.2,0.7,r"$RMSE =$"+"{0:.4f}".format(rmss),fontsize = 7)
plt.title('Datos Observados Simulados WRF')
plt.savefig("Qollpana_comparacionv2.png",dpi=300)
# dfDist.to_csv("Qollpana_comparacion.csv")
plt.show() 
plt.close()
# dfstat = pd.DataFrame(StatData,columns= ['Fecha','V mean', 'V std','c', 'k','RWeibull', 'RRayleigh'])
# print(dfstat.head())

# with pd.ExcelWriter("StatMonths/StatResumen.xlsx") as writer:
#     dfDist.to_excel(writer, sheet_name='Sheet1')
count, bins, ignored = plt.hist(dfauxs['W10s_3km'],bins=range(0,390,30),label='Datos WRF',alpha=0.2) 
plt.hist(dfaux['Viento - Direccion (°)'],bins=range(0,390,30),label='Datos Observados') 
#   Calculando el porcentaje de Datos Acumulados
Ener= (count)/sum(count)

plt.xlabel("Direccion del Viento (°)")
plt.ylabel("Distribution")
plt.title("Distribucion Direccion de Viento") 
# plt.savefig('Distribucion12Direccion.png', dpi =300)
legend()
# plt.savefig('DistribucionDireccion.png', dpi =300)
plt.show()
plt.close()


# auxt = df["Viento - Velocidad (m/s)"]*df["Viento - Velocidad (m/s)"]*df["Viento - Velocidad (m/s)"]
# aux2t = sum(auxt) / df.shape[0]


# aux = dfaux["Viento - Velocidad (m/s)"]*dfaux["Viento - Velocidad (m/s)"]*dfaux["Viento - Velocidad (m/s)"]
# aux2 = sum(aux) / dfaux.shape[0]

# Intervalo = ["0-30","30-60", "60-90", "90-120", "120-150", "150-180","180-210", "210-240", "240-270", "270-300", 
#                         "300-330","330-360"]
# StatData = {    'Sector': Intervalo,
#                     'Porcentaje de Datos Agrupados' : Ener
#             }
# dfDist = pd.DataFrame(StatData, columns = ['Sector', 'Porcentaje de Datos Agrupados'])
