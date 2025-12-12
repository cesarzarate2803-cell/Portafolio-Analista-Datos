# Análisis Predictivo de Rotación de Empleados (HR Analytics)

## Objetivo del Proyecto
Desarrollar un modelo de Machine Learning para predecir qué empleados tienen mayor riesgo de abandonar la empresa, identificando los factores clave que influyen en la rotación laboral (attrition) y proporcionando insights accionables para el departamento de Recursos Humanos.

## Herramientas y Tecnologías Utilizadas

### Lenguajes y Librerías
- **Python 3.13**
  - `pandas` - Manipulación y análisis de datos
  - `numpy` - Operaciones numéricas
  - `matplotlib` - Visualizaciones básicas
  - `seaborn` - Visualizaciones estadísticas avanzadas
  - `scikit-learn` - Machine Learning y evaluación de modelos

### Entorno de Desarrollo
- **VS Code** - Editor de código
- **Entorno Virtual** (.venv) - Gestión de dependencias

## Dataset
- **Fuente**: IBM HR Analytics Employee Attrition & Performance (Kaggle)
- **Registros**: 1,470 empleados
- **Variables**: 35 columnas
- **Período**: Datos históricos de empleados
- **Variable Objetivo**: Attrition (Yes/No)

### Variables Clave Analizadas
- Demográficas: Edad, Género, Estado Civil, Distancia desde Casa
- Laborales: Departamento, Rol, Años en la Empresa, Salario Mensual
- Satisfacción: Work-Life Balance, Satisfacción Laboral, Ambiente de Trabajo
- Compensación: Salario Mensual, Incremento Salarial, Nivel de Stock Options

## Metodología del Proyecto

### **FASE 1: Análisis Exploratorio de Datos (EDA)**
Exploración inicial para entender la distribución y calidad de los datos.

**Hallazgos principales:**
- No hay valores nulos - datos completos
- 16.12% de tasa de attrition general (237 empleados)
- Dataset balanceado demográficamente

**Análisis realizado:**
```python
# Estadísticas descriptivas
- Distribución de attrition por departamento, rol, género
- Análisis de edad, salario y distancia vs attrition
- Evaluación de work-life balance y satisfacción
```

### **FASE 2: Visualización de Datos**
Creación de 8 gráficos profesionales para identificar patrones.

**Visualizaciones generadas:**
1. Distribución general de Attrition (barras + pastel)
2. Tasa de Attrition por Departamento
3. Top 10 Roles con Mayor Attrition
4. Distribución de Salario por Attrition (boxplot)
5. Distribución de Edad por Attrition (histograma)
6. Distancia desde Casa vs Attrition (violin plot)
7. Work-Life Balance vs Attrition
8. Mapa de Calor de Correlaciones

### **FASE 3: Machine Learning - Modelo Predictivo**
Implementación de Random Forest Classifier para predecir attrition.

**Preparación de datos:**
```python
# Encoding de variables categóricas
- 7 variables categóricas convertidas a numéricas con LabelEncoder
- División: 80% entrenamiento, 20% prueba (stratified)
- 30 features finales para el modelo
```

**Configuración del modelo:**
```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    random_state=42
)
```

## Resultados del Modelo

### Métricas de Desempeño
------------------------------------------------------------------------------------
|           Métrica            |  Valor |             Interpretación               |
|------------------------------|--------|------------------------------------------|
| **Accuracy**                 | 83.67% | Muy bueno para clasificación binaria     |
| **ROC-AUC**                  | 0.82   | Excelente capacidad discriminativa       |
| **Precisión (No Attrition)** | 0.85   | Alta confiabilidad                       |
| **Recall (No Attrition)**    | 0.97   | Detecta muy bien empleados que se quedan |
| **F1-Score**                 | 0.84   | Balance general sólido                   |
------------------------------------------------------------------------------------

### Matriz de Confusión
```
                Predicho No    Predicho Yes
Real No            240              7
Real Yes            41              6

Clasificaciones correctas: 246/294 (83.67%)
```

### Top 10 Variables Más Importantes

El modelo identificó los factores que más influyen en la predicción de attrition:
---------------------------------------------------------------------------------
| # |     Variable   | Importancia |                 Insight                    |
|---|-------------------|-------|-----------------------------------------------|
| 1 | MonthlyIncome     | 8.80% | Factor #1: Salarios bajos aumentan riesgo     |
| 2 | TotalWorkingYears | 8.16% | Empleados con menos experiencia se van más    |
| 3 | Age               | 7.64% | Empleados jóvenes tienen mayor rotación       |
| 4 | OverTime          | 5.42% | Horas extras incrementan probabilidad de irse |
| 5 | DailyRate         | 4.99% | Compensación diaria afecta retención          |  
| 6 | DistanceFromHome  | 4.73% | Vivir lejos del trabajo aumenta riesgo        |
| 7 | MonthlyRate       | 4.62% | Estructura salarial mensual importa           |
| 8 | YearsAtCompany    | 4.49% | Menos antigüedad = mayor riesgo               |
| 9 | HourlyRate        | 4.04% | Tasa horaria influye en decisión              |
| 10 | JobRole          | 3.63% | Ciertos roles tienen más rotación             |
---------------------------------------------------------------------------------
## Insights Clave y Hallazgos

### Factores de Alto Riesgo

#### 1. **Departamento SALES - CRÍTICO**
- **Attrition: 20.63%** (vs 13.84% en R&D)
- Sales tiene 50% más rotación que otros departamentos

#### 2. **Sales Representative - CRISIS**
- **Attrition: 39.76%** ← 4 de cada 10 se van
- El rol con mayor rotación de toda la empresa
- Requiere intervención inmediata

#### 3. **Perfil de Riesgo: "Empleado Joven y Mal Pagado"**
- **Edad promedio de quienes se van**: 33.6 años (vs 37.6 que se quedan)
- **Salario promedio de quienes se van**: $4,787 (vs $6,833 que se quedan)
- **Diferencia salarial**: -30% menos que empleados retenidos

#### 4. **Distancia desde Casa**
- Quienes se van viven **10.6 km** del trabajo (vs 8.9 km)
- 1.7 km adicionales correlacionan con mayor attrition

#### 5. **Work-Life Balance MALO**
- Empleados con balance "Bad" (nivel 1): **31.25% de attrition**
- Empleados con balance "Better/Best": **14-17% de attrition**
- **Doble probabilidad** de irse con mal balance

### Factores de Protección

- **Edad >35 años** - Más estables
- **Salario >$6,000** - Mayor retención
- **Distancia <7 km** - Facilita permanencia
- **Roles gerenciales** - Solo 4.9% de attrition
- **Work-Life Balance nivel 3-4** - Empleados satisfechos

## Recomendaciones Estratégicas para RRHH

### **PRIORIDAD ALTA - Acción Inmediata**

1. **Intervención en Sales Representatives**
   - Revisión salarial urgente (actualmente $2,046 por debajo del promedio)
   - Programa de retención específico para este rol
   - Análisis de carga laboral y horas extras

2. **Programa de Retención para Empleados Jóvenes (<35 años)**
   - Plan de carrera claro con milestones
   - Mentoría con empleados senior
   - Incremento salarial competitivo

3. **Política de Work-Life Balance**
   - Reducir horas extras obligatorias
   - Implementar trabajo híbrido/remoto
   - Revisar empleados con nivel 1-2 de balance

### **PRIORIDAD MEDIA - Corto Plazo (3-6 meses)**

4. **Revisión Salarial por Departamento**
   - Benchmark salarial vs mercado
   - Ajuste para roles con alta rotación
   - Transparencia en bandas salariales

5. **Opciones de Trabajo Remoto/Híbrido**
   - Prioridad para empleados que viven >10 km
   - Reducir impacto de distancia en retención
   - Política flexible por departamento

6. **Dashboard Predictivo de Attrition**
   - Implementar modelo en producción
   - Alertas mensuales de empleados en riesgo
   - Seguimiento de efectividad de intervenciones

### **PRIORIDAD BAJA - Largo Plazo (6-12 meses)**

7. **Cultura Organizacional**
   - Encuestas de satisfacción trimestrales
   - Programas de reconocimiento
   - Mejora del ambiente laboral (según EnvironmentSatisfaction)

8. **Desarrollo Profesional**
   - Capacitaciones y certificaciones
   - Rotación interna de roles
   - Planes de sucesión claros

## Estructura del Proyecto
```
Proyecto-2-HR-Analytics/
│
├── data/
│   └── HR-Employee-Attrition.xlsx          # Dataset original
│
├── outputs/
│   ├── 01_attrition_distribution.png       # Gráfico: Distribución general
│   ├── 02_attrition_by_department.png      # Gráfico: Por departamento
│   ├── 03_top10_roles_attrition.png        # Gráfico: Top roles críticos
│   ├── 04_salary_vs_attrition.png          # Gráfico: Salario vs attrition
│   ├── 05_age_distribution.png             # Gráfico: Edad vs attrition
│   ├── 06_distance_vs_attrition.png        # Gráfico: Distancia vs attrition
│   ├── 07_worklifebalance_attrition.png    # Gráfico: WLB vs attrition
│   ├── 08_correlation_heatmap.png          # Gráfico: Mapa de correlaciones
│   ├── 09_confusion_matrix.png             # Gráfico: Matriz confusión ML
│   ├── 10_feature_importance.png           # Gráfico: Variables importantes
│   ├── 11_roc_curve.png                    # Gráfico: Curva ROC
│   ├── feature_importance.csv              # Ranking completo de variables
│   └── predictions.csv                     # Predicciones del modelo
│
├── hr_analysis.py                          # Script: Análisis exploratorio
├── hr_visualizations.py                    # Script: Generación de gráficos
├── hr_machine_learning.py                  # Script: Modelo predictivo
│
└── README.md                               # Este archivo
```

## Cómo Ejecutar el Proyecto

### Requisitos Previos
```bash
Python 3.8+
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
```

### Ejecución
```bash
# 1. Análisis exploratorio
python hr_analysis.py

# 2. Generar visualizaciones
python hr_visualizations.py

# 3. Entrenar modelo ML
python hr_machine_learning.py
```

## Habilidades Demostradas

- **Python Avanzado**: Pandas, NumPy, Matplotlib, Seaborn
- **Análisis Exploratorio de Datos (EDA)**: Identificación de patrones
- **Visualización de Datos**: 11 gráficos profesionales en HD
- **Machine Learning**: Random Forest Classifier, Feature Engineering
- **Evaluación de Modelos**: Métricas, Matriz de Confusión, ROC-AUC
- **Feature Importance**: Identificación de variables críticas
- **Storytelling con Datos**: Traducción de análisis a insights accionables
- **Pensamiento Estratégico**: Recomendaciones basadas en datos

*Proyecto desarrollado como parte de mi portafolio profesional de Análisis de Datos y Machine Learning*

**Última actualización**: Diciembre 2025
