from pylab import *
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error
import seaborn as sns
register_matplotlib_converters()
#******************************************************************
#					Datos Observados
#******************************************************************
Archivo = '/home/opti3040a/Documentos/WRF/Qollpana150914-270818.csv'
# Archivo = '2017_2018Q_UTC.csv'
df = pd.read_csv(Archivo, index_col= False)
df["Fecha"] = pd.to_datetime(df["Fecha"])
df['Fecha'] = df['Fecha'] + pd.Timedelta(hours=4)   # Corrigiendo UTC-4
df.to_csv('Qollpana150914-270818UTC.csv')
mask = df.Fecha.dt.year == 2018
dfvar = df.loc[mask]
mask = dfvar.Fecha.dt.month == 3
dfaux = dfvar.loc[mask]
print(dfaux.head())

#******************************************************************
#		          Filtrando los datos cada hora
#******************************************************************
# Filtrando Fecha cada hora
Int_10 = dfaux.Fecha.dt.minute == 0
df2 = dfaux.loc[Int_10].reset_index().drop(['index','Unnamed: 0','Viento - Rafaga (m/s)','Precipitation (mm)','Presion Barometrica (hPa)','Viento - Desviacion Estandar (m/s)','Humedad Relativa (% RH)'], axis=1)
# df2 = dfaux.loc[Int_10].reset_index()
dfaux = df2             # Datos observados de Marzo cada hora
# Ordenando los datos
dfaux = dfaux[['Fecha', 'Temperatura (°C)','Viento - Velocidad (m/s)','Viento - Direccion (°)']]
# Cambiando los nombres de las columnas
dfaux.columns = ["Time","T2o","W10o","Wdo"]
print(dfaux.head())
#******************************************************************
#					Datos Simulados Cada hora Marzo
#******************************************************************
Archivo = 'df_marzoCor2.csv'
dfs = pd.read_csv(Archivo, index_col= False)
dfauxs = dfs.drop(['Unnamed: 0'], axis=1)
dfauxs.columns = ["Time","T2_3km","W10_3km","Wd_3km"]
# Reemplazando '_' por ' '
dfauxs['Time'] = dfauxs['Time'].str.replace("_",' ')
dfauxs["Time"] = pd.to_datetime(dfauxs["Time"])
print(dfauxs.tail())
#******************************************************************
#		Comparativa dfauxs(simulado) vs dfaux(Observado)
#******************************************************************
# Generando solo un Dataframe para datos simulados y observados
df_Total = dfaux.merge(dfauxs,on=["Time"])
print(df_Total.head())
print(df_Total.shape)
# Calculando coeficiente de correlacion
pearson_coef, p_value = stats.pearsonr(df_Total['W10o'], df_Total['W10_3km'])
print("coeficiente de correlacion Pearson es:",pearson_coef)
# Comparando V10o vs V10WRF
plt.subplots_adjust(hspace=0.8)
plt.subplot(2,1,1)
W10o = df_Total['W10o'].tolist()
W10wrf = df_Total['W10_3km'].tolist()
Fecha = df_Total['Time'].tolist()
plt.plot_date(Fecha,W10o,'-r')
plt.plot_date(Fecha,W10wrf,'--b')
plt.title("Comparativa Marzo 2018")
plt.ylabel('Velocidad de Viento m/s')
plt.xticks(rotation=45)
# Graficando la Correlacion
plt.subplot(2,1,2)
sns.regplot(x="W10o", y="W10_3km",data=df_Total)
plt.title("Correlacion Viento Observado vs Simulado")
plt.text(min(W10o)*0.2,max(W10wrf)*0.8,r"$r^2 =$"+"{0:.4f}".format(pearson_coef),fontsize = 10, color = 'blue')
plt.savefig('EstadisticosMarzoCor.png',dpi=300)
plt.show()
plt.close()
