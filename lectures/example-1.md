# Prompt
```
<task>
Rewrite the following e-mail. 
</task>
<instructions>
Be polite, direct, and concise. State what you need in the first two sentences. Sign your e-mail with your name and student number. 
</instructions>
<email>
Hi teacher, 
I was coming to school today but the bus broke down. I was so stressed. I tried to find other busses but the road was closed. 
The weather was also bad. So, I missed our lecture. My name is Ali. I don't know any body from the class either. I started one 
semester late. I am an "irregular". Can you send me your class notes? Is there any place I can get them. This would be 
super helpful. 

See you next week teacher.
</email>
```

# Response 


    Hello Teacher,

    I unfortunately missed today's lecture due to transportation issues. I would be grateful if you could provide me with a copy of the class notes. Is there a central location where I can access them?

    Thank you for your understanding.

    Sincerely,
	Ali
	[Student Number]


# Prompt \# 1

```
<task>
Explain what the following block of python code does with the context given below. 
</task>
<instructions>
Be direct. Do not simplify the language.  The audience knows the mathematical theory 
behind the code. So, provide mathematical details with formulas if necessary.
</instructions>
<code>
    X = np.array([1, 2, 4, 6, 8, 10], dtype=float)
    Y = np.array([3, 5, 7, 10, 11, 14], dtype=float)
    mean_x = X.mean()
    mean_y = Y.mean()
    m = ((X - mean_x) * (Y - mean_y)).sum() / np.power(X - mean_x, 2).sum()
    b = mean_y - m * mean_x	
</code>
<context>
You are a senior programmer and data scientist working at a data science startup. You have
several junior data scientist working under you, and you are helping them to understand
company's code base. Your task is to write protype code for a task, and they need to refactor
your code. They need to rewrite the code to make it more efficient and then they need merge
newly written code into company's code base.
</context>
```

# Answer \# 1

This block of Python code computes the **ordinary least squares (OLS) regression coefficients** for a simple linear regression model:

$$ Y = mX + b $$

where $X$ is the independent variable and $Y$ is the dependent variable.

## Define Data

Two NumPy arrays are created:  

- $X = [1, 2, 4, 6, 8, 10]$
- $Y = [3, 5, 7, 10, 11, 14]$  

These represent paired data points \((X_i, Y_i)\).

## Compute Means

- $\bar{X} = \frac{1}{n}\sum_{i=1}^{n} X_i$
- $\bar{Y} = \frac{1}{n}\sum_{i=1}^{n} Y_i$

The mean values are used to center the data.

## Compute the Slope $m$ (Regression Coefficient)
The slope $m$ is given by the formula:

$$ m = \frac{\sum_{i=1}^{n} (X_i - \bar{X}) (Y_i - \bar{Y})}{\sum_{i=1}^{n} (X_i - \bar{X})^2} $$

In the code:
```python
m = ((X - mean_x) * (Y - mean_y)).sum() / np.power(X - mean_x, 2).sum()
```
This implements the covariance of $X$ and $Y$ divided by the variance of $X$.

## Compute the Intercept $b$
The intercept is computed as:

$$ b = \bar{Y} - m \bar{X} $$

In the code:
```python
b = mean_y - m * mean_x
```
This ensures that the regression line passes through the mean of the data.

### **Optimization Considerations**

- The computation of $m$ can be optimized by using NumPy's built-in `np.cov(X, Y, ddof=0)[0,1] / np.var(X, ddof=0)`, which is more efficient.
- The entire regression fit can be computed using `np.polyfit(X, Y, deg=1)`, which is optimized in C.

## Refactored Code:

```python
import numpy as np

X = np.array([1, 2, 4, 6, 8, 10], dtype=float)
Y = np.array([3, 5, 7, 10, 11, 14], dtype=float)

# Calculate mean once
mean = np.mean([X, Y], axis=0)

# Calculate deviations from mean once
dx, dy = X - mean[0], Y - mean[1]

# Calculate slope and intercept in one step
m, b = np.linalg.inv(np.dot(dx.T, dx)).dot(np.dot(dx.T, dy))

print("Slope:", m)
print("Intercept:", b)
```

## Improvements

- The refactored code calculates the mean and deviations from the mean only once, which improves efficiency.
- The slope and intercept are calculated in a single step using matrix operations, which is more efficient than the original code.

# Prompt 

```
Here is what the junior programmer gave: analyze the code for bugs: 
<code>
import numpy as np

X = np.array([1, 2, 4, 6, 8, 10], dtype=float)
Y = np.array([3, 5, 7, 10, 11, 14], dtype=float)

# Calculate mean once
mean = np.mean([X, Y], axis=0)

# Calculate deviations from mean once
dx, dy = X - mean[0], Y - mean[1]

# Calculate slope and intercept in one step
m, b = np.linalg.inv(np.dot(dx.T, dx)).dot(np.dot(dx.T, dy))
</code>
```

# Answer 

The junior programmer’s code has a **critical bug** in the calculation of the slope $m$ and intercept $b$. Let's analyze the issues:

---

## Identified Bugs

### 1. Incorrect Mean Calculation
```python
mean = np.mean([X, Y], axis=0)
```
- This calculates the mean across corresponding elements in `X` and `Y`, not the individual means of `X` and `Y`.
- Correct approach:
  ```python
  mean_x, mean_y = np.mean(X), np.mean(Y)
  ```

### 2. Incorrect Mean Deviations
```python
dx, dy = X - mean[0], Y - mean[1]
```
- Since `mean` is incorrectly computed as a vectorized mean across both `X` and `Y`, `mean[0]` and `mean[1]` do not represent the means of `X` and `Y`, respectively.
- Instead, they contain incorrect values due to improper broadcasting.
- Fix:
  ```python
  dx, dy = X - mean_x, Y - mean_y
  ```

### 3. Incorrect Regression Formula
```python
m, b = np.linalg.inv(np.dot(dx.T, dx)).dot(np.dot(dx.T, dy))
```

#### Mistake:  

- The slope formula in simple linear regression is:  
  $$
  m = \frac{\sum (X_i - \bar{X}) (Y_i - \bar{Y})}{\sum (X_i - \bar{X})^2}
  $$

- The above line incorrectly tries to use a matrix inversion approach meant for multiple linear regression, but in simple linear regression, $X$ is a vector, not a matrix.

- The correct formula should be:
  ```python
  m = np.sum(dx * dy) / np.sum(dx  2)
  ```
  
- Intercept $b$ is missing  

- The intercept $b$ must be explicitly calculated:
  ```python
  b = mean_y - m * mean_x
  ```

### Corrected Code 
```python
import numpy as np

X = np.array([1, 2, 4, 6, 8, 10], dtype=float)
Y = np.array([3, 5, 7, 10, 11, 14], dtype=float)

# Compute correct means
mean_x, mean_y = np.mean(X), np.mean(Y)

# Compute deviations
dx, dy = X - mean_x, Y - mean_y

# Correct slope computation
m = np.sum(dx * dy) / np.sum(dx  2)

# Correct intercept computation
b = mean_y - m * mean_x

print(f"Slope (m): {m}, Intercept (b): {b}")
```

### Hand Corrected Code (v1)
```python
import numpy as np

X = np.array([1, 2, 4, 6, 8, 10], dtype=float)
Y = np.array([3, 5, 7, 10, 11, 14], dtype=float)

# Compute correct means
mean_x, mean_y = np.mean(X), np.mean(Y)

# Compute deviations
dx, dy = X - mean_x, Y - mean_y

# Correct slope computation
m = np.dot(dx,dy) / np.dot(dx,dx)

# Correct intercept computation
b = mean_y - m * mean_x

print(f"Slope (m): {m}, Intercept (b): {b}")
```

### Hand Corrected Code (v2)
```python
import numpy as np

X = np.array([1, 2, 4, 6, 8, 10], dtype=float)
Y = np.array([3, 5, 7, 10, 11, 14], dtype=float)

# Compute correct means
mean_x, mean_y = np.mean(X), np.mean(Y)

# Correct slope computation
m = np.cov(X,Y,ddof=0)[0,1] / np.var(X,ddof=0)

# Correct intercept computation
b = mean_y - m * mean_x

print(f"Slope (m): {m}, Intercept (b): {b}")
```

### Hand Corrected Code (v3)
```python
import numpy as np

X = np.array([1, 2, 4, 6, 8, 10], dtype=float)
Y = np.array([3, 5, 7, 10, 11, 14], dtype=float)

m,b = np.polyfit(X,Y,deg=1)

print(f"Slope (m): {m}, Intercept (b): {b}")
```

# Prompt 

```
<task>
Explain in detail what this piece of code does. Provide a native pandas implementation.
</task>
<context>
You are a senior programmer and data scientist working at a data science startup. You have
several junior data scientist working under you, and you are helping them to understand
company's code base. Your task is to write protype code for a task, and they need to refactor
your code. They need to rewrite the code to make it more efficient and then they need merge
newly written code into company's code base.

Your team is asked to process a dataset provided by Istanbul Municipality data server. 
Here is sample of the provided dataset:
<data>
transition_date 	transition_hour 	transport_type_id 	road_type 	line 	transfer_type 	number_of_passage 	number_of_passenger 	product_kind 	transaction_type_desc 	town 	line_name 	station_poi_desc_cd
0 	2024-10-01 	0 	1 	OTOYOL 	CEBECI - TAKSIM 	Normal 	3 	3 	TAM 	Tam Kontur 	SARIYER 	36T 	NaN
1 	2024-10-01 	0 	1 	OTOYOL 	USKUDAR-GUZELTEPE-UMRANIYE DEVLET HASTANESI 	Aktarma 	2 	2 	INDIRIMLI1 	Indirimli Aktarma 	BAKIRKOY 	15B 	NaN
2 	2024-10-01 	0 	1 	OTOYOL 	SARIYER-HACIOSMAN-MECIDIYEKOY-TAKSIM 	Normal 	1 	1 	TAM 	Tam Abonman 	KAGITHANE 	25G 	NaN
3 	2024-10-01 	0 	1 	OTOYOL 	YESILPINAR - ALIBEYKOY METRO 	Aktarma 	1 	1 	TAM 	Tam Aktarma 	BAKIRKOY 	TM10 	NaN
4 	2024-10-01 	0 	1 	OTOYOL 	UMRANIYE DEV.HAST-CAKMAK MAH-ATASEHIR-USTBOSTANCI 	Aktarma 	1 	1 	INDIRIMLI2 	Indirimli Tip 2 Aktarma
</data>
<code>
result = db.sql("""
  select sum(number_of_passage) as total, transition_hour as hour 
     from data
     where road_type = 'RAYLI'
     group by hour
     order by hour
""").to_df()
</code>
</context>
```

# Answer 

The provided code snippet is written using DuckDB, which is an in-process SQL OLAP database management system. It's designed to handle analytical queries on datasets and is especially useful for working 
with large dataframes directly in Python.

Here’s a detailed explanation of what the code does:

1. **Database Connection**:
   - The code assumes that there is a DuckDB database instance named `db`. This instance is used to execute SQL queries against the dataset, which we assume has been loaded into this database under the 
table name `data`.

2. **SQL Query Execution**:
   - `db.sql(...)`: Executes an SQL query within the context of the connected DuckDB instance.
   
3. **SQL Query Details**:
   - The SQL query is wrapped in a string and passed to `db.sql()`. It performs several operations on the dataset:

     ```sql
     SELECT sum(number_of_passage) as total, transition_hour as hour 
     FROM data
     WHERE road_type = 'RAYLI'
     GROUP BY hour
     ORDER BY hour;
     ```
   
   - **SELECT Clause**: Specifies that we want to retrieve two columns:
     - `sum(number_of_passage) as total`: Computes the sum of the `number_of_passage` column for each group and labels it as `total`.
     - `transition_hour as hour`: Selects the `transition_hour` column and renames it as `hour`.

   - **FROM Clause**: Indicates that the data is being selected from a table named `data`.

   - **WHERE Clause**: Filters the dataset to include only rows where the `road_type` is equal to `'RAYLI'`. This means we're focusing our analysis specifically on transportation events occurring on roads 
of this type.

   - **GROUP BY Clause**: Groups the results by the `transition_hour`. For each unique hour, it calculates the sum of `number_of_passage`.

   - **ORDER BY Clause**: Orders the final output by `hour` in ascending order. This ensures that the result set is sorted chronologically by hour.

4. **Conversion to DataFrame**:
   - `.to_df()`: Converts the resulting SQL query result into a pandas DataFrame, which is stored in the variable `result`.

5. **Purpose of the Code**:
   - The overall purpose of this code block is to calculate and organize the total number of passages that occur for each hour on roads classified as `'RAYLI'`. The output is a structured DataFrame where 
each row represents an hour with its corresponding total number of passages.

6. **Pandas-Based Alternative:**

    Instead of SQL, if the data is already loaded in Pandas:

	```python
	result = (
		data[data["road_type"] == "RAYLI"]
		.groupby("transition_hour")["number_of_passage"]
		.sum()
		.reset_index()
		.sort_values("transition_hour")
	)
	```

# Prompt 

Even if we are careful in constructing our prompts, LLMs sometimes tend to *hallucinate* Consider the following exchanges:


## Prompt

```
We have a data frame at the URL https://data.ibb.gov.tr/tr/dataset/a6855ce7-4092-40a5-82b5-34cf3c7e36e3/resource/ea8daee8-62e4-4c42-903f-15b5e372d902/download/hourly_transportation_202410.csv 
How do we import the data into a DataFrame using pandas library?
```

## Answer

To import the hourly transportation data into a DataFrame using pandas, you can use the following code:

```python
import pandas as pd

# URL of the dataset
url = 'https://data.ibb.gov.tr/tr/dataset/a6855ce7-4092-40a5-82b5-34cf3c7e36e3/resource/ea8daee8-62e4-4c42-903f-15b5e372d902/download/hourly_transportation_202410.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(url)

# Print the first few rows of the DataFrame to verify the data
print(df.head())
```

This code imports the pandas library, sets the URL of the dataset, reads the CSV file into a DataFrame using `pd.read_csv()`, and prints the first few rows of the DataFrame to ensure that the data was 
imported correctly.

Please note that you may need to install the required libraries (in this case, pandas) by running `pip install pandas` in your command line before running this code.

## Prompt

```
<task>
Using duckdb, write a piece of code that filters the dataframe where the column 'road_type' is 'RAYLI'.
</task>
<context>
Here is the first 4 lines of the data frame we imported:
<data>
transition_date 	transition_hour 	transport_type_id 	road_type 	line 	transfer_type 	number_of_passage 	number_of_passenger 	product_kind 	transaction_type_desc 	town 	line_name 	station_poi_desc_cd
0 	2024-10-01 	0 	1 	OTOYOL 	CEBECI - TAKSIM 	Normal 	3 	3 	TAM 	Tam Kontur 	SARIYER 	36T 	NaN
1 	2024-10-01 	0 	1 	OTOYOL 	USKUDAR-GUZELTEPE-UMRANIYE DEVLET HASTANESI 	Aktarma 	2 	2 	INDIRIMLI1 	Indirimli Aktarma 	BAKIRKOY 	15B 	NaN
2 	2024-10-01 	0 	1 	OTOYOL 	SARIYER-HACIOSMAN-MECIDIYEKOY-TAKSIM 	Normal 	1 	1 	TAM 	Tam Abonman 	KAGITHANE 	25G 	NaN
3 	2024-10-01 	0 	1 	OTOYOL 	YESILPINAR - ALIBEYKOY METRO 	Aktarma 	1 	1 	TAM 	Tam Aktarma 	BAKIRKOY 	TM10 	NaN
4 	2024-10-01 	0 	1 	OTOYOL 	UMRANIYE DEV.HAST-CAKMAK MAH-ATASEHIR-USTBOSTANCI 	Aktarma 	1 	1 	INDIRIMLI2 	Indirimli Tip 2 Aktarma
</data>
</context>
```

## Answer

Here's an example of how you can filter the DataFrame using DuckDB:

```sql
-- Create a connection to the dataset
CREATE TABLE hourly_transportation AS SELECT * FROM @hourly_transportation_202410;

-- Filter the DataFrame where 'road_type' is 'RAYLI'
SELECT * FROM hourly_transportation WHERE road_type = 'RAYLI';
```

However, since we're using pandas and DuckDB's python interface (pyduckdb), we can use the following code:

```python
import pyducksql

# Create a connection to the dataset
conn = pyducksql.connect('https://data.ibb.gov.tr/tr/dataset/a6855ce7-4092-40a5-82b5-34cf3c7e36e3/resource/ea8daee8-62e4-4c42-903f-15b5e372d902/download/hourly_transportation_202410.csv')

# Filter the DataFrame where 'road_type' is 'RAYLI'
df_filtered = conn.execute('SELECT * FROM hourly_transportation WHERE road_type = \'RAYLI\'')

# Print the filtered DataFrame
print(df_filtered)
```

This code connects to the dataset, filters the DataFrame using DuckDB's SQL syntax, and prints the filtered results.

### Bingo!

We got a false answer. This code looks legitimate but it is completely wrong. For this example I used `llama3.2`.

Here is ChatGPT with the same instructions:

## Prompt

```
We have a data frame at the URL https://data.ibb.gov.tr/tr/dataset/a6855ce7-4092-40a5-82b5-34cf3c7e36e3/resource/ea8daee8-62e4-4c42-903f-15b5e372d902/download/hourly_transportation_202410.csv 
How do we import the data into a DataFrame using pandas library?
```

## Answer

To import the CSV data from the provided URL into a Pandas DataFrame, you can use the following Python code:

``` python
import pandas as pd

url = "https://data.ibb.gov.tr/tr/dataset/a6855ce7-4092-40a5-82b5-34cf3c7e36e3/resource/ea8daee8-62e4-4c42-903f-15b5e372d902/download/hourly_transportation_202410.csv"
df = pd.read_csv(url)

# Display the first few rows
print(df.head())
```

## Prompt

```
<task>
Using duckdb, write a piece of code that filters the dataframe where the column 'road_type' is 'RAYLI'.
</task>
<context>
Here is the first 4 lines of the data frame we imported:
<data>
transition_date     transition_hour     transport_type_id   road_type   line    transfer_type   number_of_passage   number_of_passenger     product_kind    transaction_type_desc   town    line_name   station_poi_desc_cd
0   2024-10-01  0   1   OTOYOL  CEBECI - TAKSIM     Normal  3   3   TAM     Tam Kontur  SARIYER     36T     NaN
1   2024-10-01  0   1   OTOYOL  USKUDAR-GUZELTEPE-UMRANIYE DEVLET HASTANESI     Aktarma     2   2   INDIRIMLI1  Indirimli Aktarma   BAKIRKOY    15B     NaN
2   2024-10-01  0   1   OTOYOL  SARIYER-HACIOSMAN-MECIDIYEKOY-TAKSIM    Normal  1   1   TAM     Tam Abonman     KAGITHANE   25G     NaN
3   2024-10-01  0   1   OTOYOL  YESILPINAR - ALIBEYKOY METRO    Aktarma     1   1   TAM     Tam Aktarma     BAKIRKOY    TM10    NaN
4   2024-10-01  0   1   OTOYOL  UMRANIYE DEV.HAST-CAKMAK MAH-ATASEHIR-USTBOSTANCI   Aktarma     1   1   INDIRIMLI2  Indirimli Tip 2 Aktarma
</data>
</context>
```

## Answer

Here’s how you can filter the DataFrame using DuckDB for rows where road_type is 'RAYLI':

``` python
import pandas as pd
import duckdb

# Load the data
url = "https://data.ibb.gov.tr/tr/dataset/a6855ce7-4092-40a5-82b5-34cf3c7e36e3/resource/ea8daee8-62e4-4c42-903f-15b5e372d902/download/hourly_transportation_202410.csv"
df = pd.read_csv(url)

# Use DuckDB to filter the DataFrame
query = "SELECT * FROM df WHERE road_type = 'RAYLI'"
filtered_df = duckdb.query(query).to_df()

# Display the filtered data
print(filtered_df.head())
```

## Prompt

```
Write a piece of code using duckdb that filters the dataframe where the column 'road_type' is 'RAYLI' and group by the column transition_hour. Then the code should sum up number_of_passage and finally display the results by sorting the results with respect to transition_hour.
```

## Answer

Here’s a complete code snippet using DuckDB to filter, group, sum, and sort the DataFrame as specified:

``` python
import pandas as pd
import duckdb

# Load the data
url = "https://data.ibb.gov.tr/tr/dataset/a6855ce7-4092-40a5-82b5-34cf3c7e36e3/resource/ea8daee8-62e4-4c42-903f-15b5e372d902/download/hourly_transportation_202410.csv"
df = pd.read_csv(url)

# DuckDB query to filter, group, sum, and sort
query = """
SELECT 
    transition_hour, 
    SUM(number_of_passage) AS total_passages
FROM df
WHERE road_type = 'RAYLI'
GROUP BY transition_hour
ORDER BY transition_hour
"""

# Execute the query
result_df = duckdb.query(query).to_df()

# Display the result
print(result_df)
```
