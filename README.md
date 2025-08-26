# Quax

A JAX-based quantized training framework that exports directly to TensorFlow Lite for efficient deployment on microcontrollers and embedded devices.

## Overview

Quax is a training and deployment framework specifically designed for resource-constrained hardware that ingests quantized flatbuffers directly.

## Key Features

- **Layer-level quantization control** - Precise control over quantization at individual layer granularity
- **Direct TFLite export** - export to flatbuffer format without intermediate conversions  
- **TensorFlow-independent** - Pure JAX implementation with no TensorFlow dependencies
- **Microcontroller-optimized** - Built specifically for deployment on embedded systems

## Installation

Install Quax via pip:

```bash
cd quax
pip install .
```

## Quick Start

Run the example model:

```bash
python3 quax_e2e_model.py
```

