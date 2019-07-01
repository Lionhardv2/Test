import pandas as pd
import numpy as np

df = pd.read_csv("TablasEs/StatResumen.csv")
meses = ['enero','febrero','marzo','abril','mayo','junio','julio','agosto','septiembre','octubre','noviembre','diciembre']
#******************************************************************
#				Eliminando las horas de la Fecha
#******************************************************************
a = df['Fecha'].to_list()
lista = a[0][0:11]
for i, lista in enumerate(a):
    lista = lista[0:11]
    a[i] = lista
print(a)
df['Fecha'] = a
del df['Unnamed: 0']
print(df)
print(df.columns)
with pd.ExcelWriter("StatResumen.xlsx") as writer:
    df.to_excel(writer, sheet_name='Sheet1')
df["Fecha"] = pd.to_datetime(df["Fecha"])
#******************************************************************
#				Convertir Meses  a literal
#******************************************************************
for i,m in enumerate(meses):
    mask = df.Fecha.dt.month == i+1
    print(i+1)
    print(m)
    #print(mask)
    mask.replace(True, np.nan, inplace = True)
    print(mask)

print(df)