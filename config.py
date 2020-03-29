from pyspark.conf import SparkConf

mode = "lucas"

if mode == "lucas":
    SPARK_CONF = SparkConf().setAll([('spark.executor.memory', '4g'), 
        ('spark.driver.memory','4g'), 
        ('spark.local.dir', '/home/lucas/.sparktmp')])
elif mode == "vincent":
    SPARK_CONF = SparkConf().setAll([('spark.executor.memory', '8g'),
                           ('spark.driver.memory','8g')])
else: 
    SPARK_CONF = SparkConf()