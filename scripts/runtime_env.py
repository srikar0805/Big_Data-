"""
Shared local runtime configuration for Spark/Ray scripts.

Spark needs a JVM and also needs a resolvable local host. Cloud/lab machines
often have hostnames that do not resolve through DNS, so we pin Spark local
mode to localhost unless the user has already provided explicit settings.
"""
import os
from pathlib import Path


def project_root(script_file):
    return Path(script_file).resolve().parents[1]


def configure_spark_runtime(root):
    java_candidates = [
        os.environ.get("JAVA_HOME"),
        str(root / ".jdk"),
        str(Path.home() / ".local" / "jdk-17"),
        "/usr/lib/jvm/java-17-openjdk-amd64",
    ]
    for java_home in java_candidates:
        if java_home and (Path(java_home) / "bin" / "java").exists():
            os.environ["JAVA_HOME"] = java_home
            os.environ["PATH"] = str(Path(java_home) / "bin") + os.pathsep + os.environ.get("PATH", "")
            break

    os.environ.setdefault("SPARK_LOCAL_HOSTNAME", "localhost")
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
