from setuptools import setup
import os
from setuptools.command.build_py import build_py
import subprocess

class BuildWithFlatc(build_py):
    def run(self):
        # Run flatc command before building
        subprocess.check_call(["flatc", "--filename-suffix", "_py_generated", "-o", "quax", "--python","--gen-onefile","--gen-object-api", "schemas/schema.fbs"])
        super().run()

setup(
    cmdclass={
        'build_py': BuildWithFlatc,
    },
)
