# Data Analytics and Data Science Projects

Этот репозитроий предназначен для проектов, выполненных во время в обучения в Яндекс.Практикум на 9-месячном интенсивнойм практическом курсе профессиональной подготовки по специальности Data Science. 

Проекты находятся в трех основных областях:
- **Исследовательский анализ данных (EDA), статистический анализ и визуализация** с использованием библиотек Python **Pandas, Matplotlib, Numpy, Seaborn, Scipy, Statsmodels** (например, выпускной проект - анализ и предсказание склонности клиента к уходу из телеком-компании; исследование рынка недвижимости Санкт-Петербурга; анализ факторов, влияющих на успех продаж видеоигр);
- **Машинное обучение** с использованием библиотек Python **Sklearn, CatBoost, LightGBM** (например, выпускной проект - анализ и предсказание склонности клиента к уходу из телеком-компании; предсказание концентрации золота в процессе обогащения; прогноз заказов такси в следующий час);
- **Deep Learning** с использованием библиотек Python **TensorFlow, Keras, spaCy, NLTK** (например, определение возраста покупателя по фотографии; определение токсичных комментариев).

## 00. Graduation project 
Определение вероятности ухода клиентов из телеком компании и максимизация выручки.

**Цели проекта:**  
1. Провести анализ данных и разработать модель машинного обучения для определения клиентов, склонных уйти из компании в ближайшее время (задача классификации), что позволит предложить ему специальные условия и избежать падения клиентской базы, выручки и прибыли компании.
2. Предложить механизм расчета влияния предложения скидок конкретным клиентам, вероятности ухода которых предсказала модель, на выручку компании с целью ее максимизации.
3. Составить портрет клиента, склонного к уходу, для дальнейшего маркетингового и продуктового анализа.

**1. Использованные инструменты:** Matplotlib, NumPy, Pandas, Python, SciPy, Scikit-learn, CatBoost. 	
**2. Действия / области:** обработка данных, пропуски, группировка, сортировка, исследовательский анализ данных, описательная статистика, проверка статистических гипотез, Машинное обучениеб настройка порога классификации для максимизации выручки.

## Project 01 
Исследование данных сервиса “Яндекс.Музыка” — сравнение пользователей двух городов.	 

На реальных данных Яндекс.Музыки c помощью библиотеки Pandas и её возможностей проверены данные и проведено сравнение поведения и предпочтений пользователей двух столиц — Москвы и Санкт-Петербурга. 
**1. Использованные инструменты:** Pandas, Python.	
**2. Действия / области:** обработка данных, дубликаты, пропуски, логическая индексация, группировка, сортировка.

## Project 02 
Исследование надёжности заёмщиков — анализ банковских данных с целью вывления факторов, влияющий на факт возврата кредита в срок.	

На основе данных кредитного отдела банка исследовано влияние семейного положения и количества детей на факт погашения кредита в срок. Была получена информация о данных. Определены и обработаны пропуски. Заменены типы данных на соответствующие хранящимся данным. Удалены дубликаты. Категоризованы данные. Один датафрейм декомпозирован на три.
**1. Использованные инструменты:** Pandas, Python, предобработка данных.	 
**2. Действия / области:** обработка данных, дубликаты, пропуски, категоризация, декомпозиция.

## Project 03 
Анализ рынка недвижимости с целью вывления факторов, влияющих на рыночную стоимость квартир.

На основе данных сервиса Яндекс.Недвижимость определены факторы, влияющие на рыночную стоимость объектов недвижимости разного типа, типичные параметры квартир, в зависимости от удаленности от центра. Проведена предобработка данных. Добавлены новые данные. Построены гистограммы, боксплоты, диаграммы рассеивания.
**1. Использованные инструменты:** Matplotlib, Pandas, Python, визуализация .
**2. Действия / области:** обработка данных, histogram, boxplot, scattermatrix, категоризация, scatterplot.

## Project 04 
Определение выгодного тарифа для телеком компании.  

На основе данных клиентов оператора сотовой связи проанализировано поведение клиентов и проведен поиск оптимального тарифа.	Проведен предварительный анализ использования тарифов на выборке клиентов, проанализировано поведение клиентов при использовании услуг оператора и рекомендованы оптимальные наборы услуг для пользователей. Проведена предобработка данных, их анализ. Проверены гипотезы о различии выручки абонентов разных тарифов и различии выручки абонентов из Москвы и других регионов.
**1. Использованные инструменты:** Matplotlib, NumPy, Pandas, Python, SciPy, описательная статистика, проверка статистических гипотез
**2. Действия / области:** обработка данных, histogram, boxplot, статистический тест, критерий Стьюдента

## Project 05 
Определения закономерностей, влияющих на успешность продаж видеоигр.

Из открытых источников доступны исторические данные о продажах видеоигр, оценки пользователей и экспертов, жанры и платформы (например, Xbox или PlayStation). Выявлены определяющие успешность игры закономерности.

**1. Использованные инструменты:** Matplotlib, Seaborn, NumPy, Pandas, Python, SciPy, описательная статистика, проверка статистических гипотез
**2. Действия / области:** обработка данных, histogram, boxplot, статистический тест, критерий Стьюдента
	
## Project 06 
Прогнозирование оттока клиента Банка.

На основе данных из банка нужно спрогнозировать, уйдёт клиент из банка в ближайшее время или нет. Предоставлены исторические данные о поведении клиентов и расторжении договоров с банком.
**1. Использованные инструменты:** Matplotlib, Pandas, Scikit-learn, Машинное обучение.
**2. Действия / области:** классификация, подбор гиперпараметров, выбор модели МОls.

## Project 07
Классификаиция клиентов телеком компании.

Оператор мобильной связи выяснил: многие клиенты пользуются архивными тарифами. Построена модель, способная на основании поведения клиентов предложить пользователям один из новых тарифов.
**1. Использованные инструменты:** Matplotlib, Pandas, Python, Scikit-learn.
**2. Действия / области:** классификация, подбор гиперпараметров, выбор модели МО.	

## Project 08 
Построение модели определения стоимости автомобиля.

На основе исторические данных построена модель для определения стоимости автомобиля.
**1. Использованные инструменты:** Pandas, Python, lightgbm.
**2. Действия / области:** градиентный бустинг, регрессия.
	
## Project 09
Определение наиболее выгодного региона нефтедобычи.

На основе данных геологоразведки необходимо выбрать район добычи нефти. Характеристики для каждой скважины в трех регионах уже известны. Построена модель для определения региона, где добыча принесёт наибольшую прибыль.
**1. Использованные инструменты:** Pandas, Scikit-learn, бутстреп.
**2. Действия / области:** регрессия, разработка бизнес-модели, бутстреп.

## Project 10
Исследование технологического процесса очистки золота.

Построена модель машинного обучения для промышленной компании, разрабатывающая решения для эффективной работы промышленных предприятий. Модель предсказывает коэффициент восстановления золота из золотосодержащей руды на основе данных с параметрами добычи и очистки.
**1. Использованные инструменты:** Matplotlib, NumPy, Pandas, Python, Scikit-learn, исследовательский анализ данных.
**2. Действия / области:** анализ данных, регрессия, кастомные метрики.
		
## Project 11
Прогнозирование количества заказов такси на следующий час.

Компания такси собрала исторические данные о заказах такси в аэропортах. Чтобы привлекать больше водителей в период пиковой нагрузки, построена модель прогнозирующая количество заказов такси на следующий час.
**1. Использованные инструменты:** Pandas, Python, Scikit-learn, statsmodels.
**2. Действия / области:** временные ряды, регрессия, предсказания.

## Project 12
Обучение модели классификации комментариев	МО для текстов.

Интернет-магазин запускает новый сервис. Теперь пользователи могут редактировать и дополнять описания товаров, как в вики-сообществах. То есть клиенты предлагают свои правки и комментируют изменения других. Разработана модель классификации, который будет искать токсичные комментарии.
**1. Использованные инструменты:** Pandas, Python, nltk, tf-idf.
**2. Действия / области:** обработка естественного языка, NLP.

## Project 13
Определение возраста покупателя.

Сетевой супермаркет внедряет систему компьютерного зрения для обработки фотографий покупателей. Фотофиксация в прикассовой зоне поможет определять возраст клиентов, чтобы анализировать покупки и предлагать товары, которые могут заинтересовать покупателей этой возрастной группы и контролировать добросовестность кассиров при продаже алкоголя. Построена модель, которая по фотографии определяет приблизительный возраст человека.
**1. Использованные инструменты:** Keras, Python.	
**2. Действия / области:** обработка изображени, нейронные сети.

## Project 14 
Защита данных клиентов страховой компании.

Разработка модели анонимизации персональных данных	Необходимо защитить данные клиентов страховой компании. Разработан метод на основе преобразования данных (на основе умножения матриц), чтобы по ним было сложно восстановить персональную информацию и чтобы при преобразовании качество моделей машинного обучения не ухудшилось. 
**1. Использованные инструменты:** NumPy, Python, Scikit-learn.
**2. Действия / области:** линейная алгебра, регрессия






							







## Part 1 - Intro to Data Analysis

Subjects Covered:
* Anaconda: Learn to use Anaconda to manage packages and environments for use with Python
* Jupyter Notebook: Learn to use this open-source web application 
* Data Analysis Process
* NumPy for 1 and 2D Data
* Pandas Series and Dataframes

### Project 1: Explore Weather Trends with weather forecast data 
In this project, I choose one of Udacity's curated datasets and investigate it using NumPy and pandas.
I complete the entire data analysis process, starting by posing a question and finishing by sharing the findings. 
( It may be better to place this section inside the readme of the project 1) 

### Project 2:  Investigate a dataset called TMDb movie data.
I was provided a dataset reflecting data collected from an experiment. I used statistical techniques to answer questions about the data and report my conclusions and recommendations in a report.



## Part 2 -Practical Statistics 

Subjects Covered:
* Probability
* Conditional Probability
* Binominal Distribution
* Sampling Distribution and Central Limit Theorem
* Descriptive Statistics
* Inferential Statistics
* Confidence Levels and Intervals
* Hypothesis Testing
* T-tests and A/B test
* Regression
* Multiple Linear Regression
* Logistic Regression

### Project 3: Analyze A/B Test Results with company ab_data.csv 
Using Python, I gathered data from a variety of sources, assess its quality and tidiness, then clean it. I documented the wrangling efforts in a Jupyter Notebook, plus showcase them through analyses and visualizations using Python and SQL.By using AB Testing and regression methods to decide if the company should launch a new webpage or keep the old one. 



## Part 3 - Data Extraction and Wrangling

Subjects Covered:
* GATHERING DATA: 
   * Gather data from multiple sources, including gathering files, programmatically downloading files, web-scraping data, and accessing data from APIs
   * Import data of various file formats into pandas, including flat files (e.g. TSV), HTML files, TXT files, and JSON files
   * Store gathered data in a PostgreSQL database
* ASSESSING DATA 
   * Assess data visually and programmatically using pandas
   * Distinguish between dirty data (content or “quality” issues) and messy data (structural or “tidiness” issues)
   * Identify data quality issues and categorize them using metrics: validity, accuracy, completeness, consistency, and uniformity
* CLEANING DATA 
   * Identify each step of the data cleaning process (defining, coding,and testing)
   * Clean data using Python and pandas
   * Test cleaning code visually and programmatically using Python

### Project 4 :  Data Wrangle and Analyze with Tweet WeRateDogs data
Collect data from different sources and assess data visually and programmatically , clean data for visulizing data and finding insights later. 


## Part 4 - Data Visualisation

Subjects Covered:
* Univariate exploration of data ( histogram , bar charts , Use axis limits and different scales ) 
* Bivariate exploration of data ( scatter plots , clustered bar charts , violin and bar charts , faceting )
* Multivariate exploration of data ( encodings , plot matrices , feature enginnering )
* Explanatory Visulizations ( story telling with data ,  polish plots , create slide deck ) 

### Project 5: Data Visulization with Diamond Data 
Data visualization to a dataset involving the characteristics of diamonds and their prices.

### Project 6: Communicate data finding with Ford Go Bike Sharing Data
In this project, I used Python’s data visualization tools to systematically explore the bike dataset for
its properties and relationships between variables. Then, I created a presentation that communicates the findings to others.