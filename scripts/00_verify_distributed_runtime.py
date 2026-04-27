"""
Verify the local distributed-computing runtime.

This script does not need the Stack Overflow survey CSV. It checks that:
  1. PySpark can start a local Spark JVM and run a small distributed query.
  2. Ray can start a local cluster and execute remote tasks.
"""
from runtime_env import configure_spark_runtime, project_root


PROJECT_ROOT = project_root(__file__)
configure_spark_runtime(PROJECT_ROOT)

from pyspark.sql import SparkSession
import ray


spark = (
    SparkSession.builder
    .appName("AI Trust Paradox - 00 Runtime Verification")
    .master("local[2]")
    .config("spark.ui.enabled", "false")
    .config("spark.sql.shuffle.partitions", "2")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

spark_sum = spark.range(10).groupBy().sum("id").first()[0]
print(f"Spark {spark.version} OK: sum(0..9) = {spark_sum}")
spark.stop()

if ray.is_initialized():
    ray.shutdown()
ray.init(num_cpus=2, include_dashboard=False, ignore_reinit_error=True, log_to_driver=False)


@ray.remote
def square(x):
    return x * x


ray_sum = sum(ray.get([square.remote(i) for i in range(5)]))
print(f"Ray {ray.__version__} OK: sum of squares 0..4 = {ray_sum}")
ray.shutdown()
