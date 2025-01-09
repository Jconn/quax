from setuptools import setup
import os
from setuptools.command.build_py import build_py
import subprocess

class BuildWithFlatc(build_py):
    def run(self):
        # Run flatc command before building
        subprocess.check_call(["flatc", "--python", "schema_v3c.fbs"])
        #move to quax
        if os.path.exists("quax/tflite"):
            subprocess.check_call(["rm", "quax/tflite", "-r"])
        subprocess.check_call(["mv", "tflite", "quax"])
        #give it an __init__.py
        subprocess.check_call(["touch", "quax/tflite/__init__.py"])

        super().run()

setup(
    cmdclass={
        'build_py': BuildWithFlatc,
    },
)
