from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

from pyspark.ml.classification import *
from pyspark.ml.clustering import *
from pyspark.ml.feature import *
from pyspark.ml.recommendation import *
from pyspark.ml.regression import *
from pyspark.ml.evaluation import *

# Create SparkSession
spark = SparkSession.builder.appName("PySpark_Test").getOrCreate()

# Load data
df = spark.read.csv("data/iris.csv", header=True, inferSchema=True)

# Show data
df.show()

# Print schema
df.printSchema()

# Print summary statistics
df.describe().show()

# Print number of rows
print("Number of rows: %d" % df.count())

# Print number of columns
print("Number of columns: %d" % len(df.columns))

# Split the data into training and test sets
(trainingData, testData) = df.randomSplit([0.7, 0.3])

# Define the features column
assembler = VectorAssembler(inputCols=df.columns[:-1], outputCol="features")

# Transform the data
assembledData = assembler.transform(trainingData)

# Create a DecisionTreeClassifier
dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")

# Fit the model
model = dt.fit(assembledData)

# Make predictions on the test data
predictions = model.transform(assembler.transform(testData))

# Evaluate the model
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy: {:.2f}%".format(accuracy * 100))

