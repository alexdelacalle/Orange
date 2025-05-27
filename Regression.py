import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns

train_df = pd.read_csv('Dataset/TRAIN.csv')
test_df = pd.read_csv('Dataset/TEST.csv')
cat_columns= ['axe_tipofamilia', 'axe_nucleo', 'axe_perfilnucleo', 'axe_perfilhijos']

y_train = train_df['TARGET'].values

cols_to_drop = ['TARGET', 'EOP_DAY', 'customer_id', 'uni_cliente_fijo', 'uni_cliente_movil', 
                'uni_cliente_segmento', 'uni_lineas_movil_cliente', 'uni_dto_valor_cli', 
                'uni_mb_consumo_movil_cli', 'uni_bono_100_mb_cli', 'uni_bono_adicional_mb_cli', 
                'uni_bono_mas_mb_cli', 'uni_bono_cesion_mb_cli', 'uni_num_llamadas_dentro_tarifa_cli', 
                'uni_minutos_dentro_tarifa_cli', 'uni_num_llamadas_fuera_tarifa_cli', 
                'uni_minutos_fuera_tarifa_cli', 'uni_importe_fuera_tarifa_cli', 'uni_ppfide_dm7', 
                'uni_flag_deposito_mb', 'uni_num_lines_pct_mb_mayor_70', 
                'uni_num_lines_pct_minutes_mayor_70', 'uni_num_lines_minutos_disponibles_ilimitados', 
                'uni_pct_lines_minutos_disponibles_ilimitados', 'uni_max_ratio_consumo_contratado_movil_mb', 
                'uni_min_ratio_consumo_contratado_movil_mb', 'uni_max_ratio_consumo_contratado_movil_minutes', 
                'uni_min_ratio_consumo_contratado_movil_minutes', 'uni_ratio_precio_consumo_mb_normalised', 
                'uni_ratio_precio_consumo_minutos_normalised', 'uni_ratio_consumo_contratado_mb', 
                'uni_ratio_consumo_contratado_minutos', 'uni_dias_ult_cdt', 'ros_duration_jazztel', 
                'ros_duration_lebara', 'ros_duration_llamaya', 'ros_duration_lowi', 
                'ros_duration_masmovil', 'ros_duration_movistar', 'ros_duration_ono', 
                'ros_duration_pepephone', 'ros_duration_republica_movil', 'ros_duration_resto_de_operadores', 
                'ros_duration_simyo', 'ros_duration_tuenti', 'ros_duration_vodafone', 
                'ros_duration_yoigo', 'trf_nro_lineas_mov_con_tarifa_top', 
                'trf_nro_lineas_mov_con_tarifa_no_top', 'trf_pct_lineas_mov_con_tarifa_no_top', 
                'trf_nro_lineas_exceso_mb', 'trf_nro_lineas_exceso_mins', 
                'trf_pct_lineas_mov_con_exceso_mb', 'trf_pct_lineas_mov_con_exceso_minutos', 
                'con_max_mb_naveg_no_util_mov_lin_dm7', 'con_min_mb_naveg_no_util_mov_lin_dm7', 
                'con_min_ratio_mb_naveg_util_mov_lin_dm7', 'con_max_ratio_mb_naveg_util_mov_lin_dm7', 
                'con_min_latencia_naveg_mov_lin_dm7', 'con_max_latencia_naveg_mov_lin_dm7', 
                'con_max_ratio_naveg_4g_mov_lin_dm7', 'con_min_ratio_naveg_4g_mov_lin_dm7', 
                'con_min_ratio_naveg_2g_mov_lin_dm7', 'con_max_ratio_naveg_2g_mov_lin_dm7', 
                'con_min_ratio_fallos_sms_mov_lin_dm7', 'con_max_ratio_fallos_sms_mov_lin_dm7', 
                'con_num_paq_retrans_naveg_mov_lin_dm7', 'con_num_paq_naveg_mov_lin_dm7', 
                'con_num_fallos_sms_env_mov_lin_dm7', 'con_num_fallos_sms_rec_mov_lin_dm7', 
                'con_num_fallos_est_llam_mov_lin_dm7', 'con_num_fallos_sms_mov_lin_dm7', 
                'con_min_retrans_naveg_mov_lin_dm7', 'con_min_ratio_llam_caidas_mov_lin_dm7', 
                'con_max_ratio_llam_caidas_mov_lin_dm7', 'con_min_ratio_llam_no_estab_mov_lin_dm7', 
                'con_max_ratio_llam_no_estab_mov_lin_dm7', 'con_min_dist_celda_home_mov_lin_dm7', 
                'con_max_dist_celda_home_mov_lin_dm7', 'con_num_llam_estab_mov_lin_dm7', 
                'con_num_sms_env_mov_lin_dm7', 'uni_riesgo', 'uni_smartphones_alta_premium', 
                'ga_dias_consulta_contrata_cdt_120d', 'ga_dias_consulta_contrata_mfp_120d', 
                'ga_dias_consulta_tarifas_cdt_120d', 'ga_dias_consulta_llaa_120d', 
                'ga_dias_consulta_tarifas_mfp_120d', 'ga_dias_consulta_dispositivo_renove_60d', 
                'ga_num_consultas_dispositivo_renove_30d', 'ga_dias_consulta_cp_120d', 
                'ga_dias_consulta_dispositivo_renove_120d']

X_train = train_df.drop(columns=cols_to_drop)

X_train = pd.get_dummies(X_train, columns=cat_columns, drop_first=True)

for col in X_train.select_dtypes(include='number').columns:
    if X_train[col].isna().any():
        X_train[col].fillna(X_train[col].median(), inplace=True)

for col in X_train.select_dtypes(include='object').columns:
    if X_train[col].isna().any():
        X_train[col].fillna(X_train[col].mode()[0], inplace=True)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = LogisticRegression(max_iter=1000, solver='lbfgs')
model.fit(X_train_scaled, y_train)

y_pred_probs = model.predict_proba(X_train_scaled)[:, 1]
y_pred = (y_pred_probs >= 0.25).astype(int)  

print("\nCoeficientes del modelo:")
for name, coef in zip(['intercepto'] + list(X_train.columns), [model.intercept_[0]] + list(model.coef_[0])):
    print(f"{name}: {coef:.4f}")



print("\nEvaluación del modelo:")
print(f"Accuracy      : {accuracy_score(y_train, y_pred):.4f}")
print(f"Precision     : {precision_score(y_train, y_pred):.4f}")
print(f"Recall        : {recall_score(y_train, y_pred):.4f}")
print(f"F1 Score      : {f1_score(y_train, y_pred):.4f}")
print("Matriz de confusión:")
print(confusion_matrix(y_train, y_pred))
coefs = model.coef_[0]
print("Número de coeficientes:",len(coefs) )
"""
plt.figure(figsize=(8, 6))
sns.boxplot(x=coefs, orient='h')
plt.title('Distribución de coeficientes del modelo')
plt.xlabel('Valor del coeficiente')
plt.grid(True)
plt.show()

p5 = np.percentile(coefs, 5)
p25 = np.percentile(coefs, 25)
p50 = np.percentile(coefs, 50)
p75 = np.percentile(coefs, 75)
p95 = np.percentile(coefs, 95)

print("Percentil  5 :", p5)
print("Percentil 25 :", p25)
print("Percentil 50 (mediana):", p50)
print("Percentil 75 :", p75)
print("Percentil 95 :", p95)
"""
coef_df = pd.DataFrame({
    'Variable': X_train.columns,
    'Coeficiente': coefs
}).sort_values(by='Coeficiente')

print("\nTop 10 coeficientes más positivos:")
print(coef_df.tail(36))

print("\nTop 10 coeficientes más negativos:")
print(coef_df.head(36))

top_36_variables = coef_df.reindex(coef_df['Coeficiente'].abs().sort_values(ascending=False).index).head(36)['Variable'].tolist()

# Exportar a archivo .npy (opcional)
np.save('top_36_variables.npy', np.array(top_36_variables))

