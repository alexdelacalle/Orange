import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
# Cargar datos
train_df = pd.read_csv('Dataset/TEST.csv')
cat_columns = ['axe_tipofamilia', 'axe_nucleo', 'axe_perfilnucleo', 'axe_perfilhijos']
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

# Rellenar NaNs
for col in X_train.select_dtypes(include='number').columns:
    if X_train[col].isna().any():
        X_train[col].fillna(X_train[col].median(), inplace=True)

for col in X_train.select_dtypes(include='object').columns:
    if X_train[col].isna().any():
        X_train[col].fillna(X_train[col].mode()[0], inplace=True)

# Escalado (opcional, RF no lo necesita realmente)
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)

# Importar las 36 variables
top_36_variables = np.load('top_36_variables.npy')

X_train_reduced = X_train[top_36_variables]

# Entrenar Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_reduced, y_train)
y_pred = rf_model.predict(X_train_reduced)

print("\nEvaluación del modelo Random Forest:")
print(f"Accuracy      : {accuracy_score(y_train, y_pred):.4f}")
print(f"Precision     : {precision_score(y_train, y_pred):.4f}")
print(f"Recall        : {recall_score(y_train, y_pred):.4f}")
print(f"F1 Score      : {f1_score(y_train, y_pred):.4f}")
print("Matriz de confusión:")
print(confusion_matrix(y_train, y_pred))

importancias = rf_model.feature_importances_

importancia_df = pd.DataFrame({
    'Variable': X_train_reduced.columns,
    'Importancia': importancias
}).sort_values(by='Importancia', ascending=False)

# Mostrar la variable más influyente
print("Variable más influyente:")
print(importancia_df.head(10))