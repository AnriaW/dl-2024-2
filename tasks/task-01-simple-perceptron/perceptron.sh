#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Por favor, forneça o número de matrícula como argumento."
    echo "Uso: $0 <número_de_matrícula>"
    exit 1
fi

REGISTRATION_NUMBER=$1

python3 perceptron.py --registration_number "$REGISTRATION_NUMBER"