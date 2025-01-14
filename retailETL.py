# Databricks notebook source
aws_access_key = dbutils.secrets.get(scope="retail", key="aws-access-key")
aws_secret_key = dbutils.secrets.get(scope="retail", key="aws-secret-access-key")
bucket_name = "retail-sales-performance"
mount_point = "/mnt/s3_mount"
file_name = "retail-sales-data.csv"
file_path = f"{mount_point}/{file_name}"

# COMMAND ----------

dbutils.fs.mount(
    source=f"s3a://{aws_access_key}:{aws_secret_key}@{bucket_name}",
    mount_point=mount_point,
)

# COMMAND ----------

dbutils.fs.ls(mount_point)

# COMMAND ----------

# MAGIC %md
# MAGIC 1. Data Ingestion: Load sales data containing product_id, sale_date, quantity_sold, and sale_amount.

# COMMAND ----------

df = spark.read.csv(file_path, header=True, inferSchema=True)

# COMMAND ----------

df.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 2. Data Cleaning:
# MAGIC ○ Filter out records with zero or negative sales.
# MAGIC

# COMMAND ----------

for col in df.columns:
    df = df.filter(df[col].isNotNull())

df = df.filter((df.quantity_sold >= 0) & (df.sale_amount >= 0))
df = df.distinct()
df.show()
   

# COMMAND ----------

print(df.count())

# COMMAND ----------

from pyspark.sql.window import Window
import pyspark.sql.functions as fx

# COMMAND ----------

# MAGIC %md
# MAGIC 3. Data Transformation:

# COMMAND ----------

# MAGIC %md
# MAGIC ○ Compute total revenue per product using sum() over a window

# COMMAND ----------

window_spec = Window.partitionBy("product_id")
revenue_per_product = df.withColumn("total_revenue_per_product", fx.sum(fx.col("quantity_sold") * fx.col("sale_amount")).over(window_spec))

revenue_per_product.show(100)

# COMMAND ----------

# MAGIC %md
# MAGIC ○ Use a window function (rank()) to rank products based on cumulative sales revenue per week.

# COMMAND ----------

weekly_revenue = revenue_per_product.withColumn("week_of_year", fx.weekofyear(fx.col("sale_date")))

# COMMAND ----------

weekly_revenue.show()

# COMMAND ----------

weekly_revenue_aggregated = (
    weekly_revenue.withColumn(
        "cumulative_sales", fx.sum("total_revenue_per_product").over(Window.partitionBy("week_of_year", "product_id", "product_name"))
        )
    )

# COMMAND ----------

weekly_revenue_aggregated.show()

# COMMAND ----------

window_spec = Window.partitionBy("week_of_year").orderBy(fx.col("cumulative_sales").desc())
df_ranked = weekly_revenue_aggregated.withColumn("rank", fx.rank().over(window_spec))
df_ranked.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 4. Data Enrichment:
# MAGIC ○ Add a column for percentile ranking of products using the percent_rank() function.
# MAGIC

# COMMAND ----------

window_spec = Window.partitionBy("week_of_year").orderBy(fx.col("cumulative_sales").desc())
df_perfect_rank = df_ranked.withColumn("percent_rank", fx.percent_rank().over(window_spec))

df_perfect_rank.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 5. Data Storage: Write the transformed dataset to a Delta table for further analytics and
# MAGIC visualization in Databricks SQL.
# MAGIC

# COMMAND ----------

delta_table_path = "s3a://{aws_access_key}:{aws_secret_key}@{bucket_name}/delta/rank"
df_perfect_rank.write.format("delta").mode("overwrite").save(delta_table_path)

# COMMAND ----------

df = spark.read.format('delta').load(f'{delta_table_path}')

# COMMAND ----------

display(df)

# COMMAND ----------

dbutils.fs.ls(mount_point)

# COMMAND ----------

dbutils.fs.unmount(mount_point)
